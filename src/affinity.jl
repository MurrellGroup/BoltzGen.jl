using Onion
using NNlib

const BGLayerNorm = Onion.BGLayerNorm

@concrete struct AffinityModule <: Onion.Layer
    boundaries
    dist_bin_pairwise_embed
    s_to_z_prod_in1
    s_to_z_prod_in2
    z_norm
    z_linear
    pairwise_conditioner
    pairformer_stack
    affinity_heads
end

@layer AffinityModule

function AffinityModule(
    token_s::Int,
    token_z::Int;
    pairformer_args::Dict=Dict(),
    transformer_args::Dict=Dict(),
    num_dist_bins::Int=64,
    max_dist::Real=22,
    use_cross_transformer::Bool=false,
    groups::Dict=Dict(),
)
    boundaries = collect(range(2f0, Float32(max_dist); length=num_dist_bins - 1))
    dist_bin_pairwise_embed = BGEmbedding(num_dist_bins, token_z; init=:zeros)

    s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
    s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

    z_norm = BGLayerNorm(token_z; eps=1f-5)
    z_linear = LinearNoBias(token_z, token_z)

    pairwise_conditioner = PairwiseConditioning(token_z, token_z; num_transitions=2)

    pairformer_stack = Onion.PairformerNoSeqModule(
        token_z,
        get(pairformer_args, :num_blocks, 1);
        dropout=Float32(get(pairformer_args, :dropout, 0.25)),
        pairwise_head_width=get(pairformer_args, :pairwise_head_width, 32),
        pairwise_num_heads=get(pairformer_args, :pairwise_num_heads, 4),
        post_layer_norm=get(pairformer_args, :post_layer_norm, false),
        activation_checkpointing=get(pairformer_args, :activation_checkpointing, false),
    )

    affinity_heads = AffinityHeadsTransformer(
        token_z,
        get(transformer_args, :token_s, token_s),
        get(transformer_args, :num_blocks, 1),
        get(transformer_args, :num_heads, 1),
        get(transformer_args, :activation_checkpointing, false),
        use_cross_transformer,
        groups=groups,
    )

    return AffinityModule(
        boundaries,
        dist_bin_pairwise_embed,
        s_to_z_prod_in1,
        s_to_z_prod_in2,
        z_norm,
        z_linear,
        pairwise_conditioner,
        pairformer_stack,
        affinity_heads,
    )
end

function (am::AffinityModule)(
    s_inputs,
    z,
    x_pred,
    feats;
    multiplicity::Int=1,
    use_kernels::Bool=false,
)
    z = am.z_linear(am.z_norm(z))
    z = repeat_interleave_batch(z, multiplicity)

    token_z = size(z, 1)
    N = size(s_inputs, 2)
    B = size(s_inputs, 3)

    z = z .+ reshape(am.s_to_z_prod_in1(s_inputs), token_z, N, 1, B) .+ reshape(am.s_to_z_prod_in2(s_inputs), token_z, 1, N, B)

    token_to_rep_atom_ff = repeat_interleave_batch(feats["token_to_rep_atom"], multiplicity)

    x_pred_ff = x_pred
    if ndims(x_pred_ff) >= 4
        tail = prod(size(x_pred_ff)[3:end])
        x_pred_ff = reshape(x_pred_ff, size(x_pred_ff, 1), size(x_pred_ff, 2), tail)
    end
    if ndims(x_pred_ff) != 3
        error("AffinityModule expected x_pred with 3 dims after flattening, got size=$(size(x_pred_ff))")
    end

    x_pred_batched = permutedims(x_pred_ff, (2, 1, 3))
    x_pred_repr = NNlib.batched_mul(token_to_rep_atom_ff, x_pred_batched)
    x_pred_repr_ff = permutedims(x_pred_repr, (2, 1, 3))

    d = pairwise_distance_batch(permutedims(x_pred_repr_ff, (3, 2, 1)))

    dist_idx = sum(reshape(d, size(d, 1), size(d, 2), size(d, 3), 1) .> reshape(am.boundaries, 1, 1, 1, :); dims=4)
    dist_idx = Int.(dropdims(dist_idx; dims=4))
    dist_idx_ff = permutedims(dist_idx, (2, 3, 1))
    distogram = am.dist_bin_pairwise_embed(dist_idx_ff)

    z = z .+ am.pairwise_conditioner(z, distogram)

    pad_token_mask_ff = repeat_interleave_batch(feats["token_pad_mask"], multiplicity)
    rec_mask_ff = Float32.(repeat_interleave_batch(feats["mol_type"], multiplicity) .== 0)
    rec_mask_ff = rec_mask_ff .* pad_token_mask_ff
    lig_mask_ff = Float32.(repeat_interleave_batch(feats["affinity_token_mask"], multiplicity) .> 0)
    lig_mask_ff = lig_mask_ff .* pad_token_mask_ff

    cross_pair_mask_ff = _unsqueeze_local(lig_mask_ff, 2) .* _unsqueeze_local(rec_mask_ff, 1) .+ _unsqueeze_local(rec_mask_ff, 2) .* _unsqueeze_local(lig_mask_ff, 1) .+ _unsqueeze_local(lig_mask_ff, 2) .* _unsqueeze_local(lig_mask_ff, 1)

    z = am.pairformer_stack(z, cross_pair_mask_ff; use_kernels=use_kernels)

    out_dict = Dict{String,Any}()
    out_heads = am.affinity_heads(z, feats; multiplicity=multiplicity)
    for (k, v) in out_heads
        out_dict[k] = v
    end
    return out_dict
end

@concrete struct AffinityHeadsTransformer <: Onion.Layer
    affinity_out_mlp
    to_affinity_pred_value
    to_affinity_pred_score
    to_affinity_logits_binary
end

@layer AffinityHeadsTransformer

function AffinityHeadsTransformer(
    token_z::Int,
    input_token_s::Int,
    num_blocks::Int,
    num_heads::Int,
    activation_checkpointing,
    use_cross_transformer;
    groups::Dict=Dict(),
)
    affinity_out_mlp = (
        BGLinear(token_z, token_z; bias=true, init=:default),
        NNlib.relu,
        BGLinear(token_z, input_token_s; bias=true, init=:default),
        NNlib.relu,
    )

    to_affinity_pred_value = (
        BGLinear(input_token_s, input_token_s; bias=true, init=:default),
        NNlib.relu,
        BGLinear(input_token_s, input_token_s; bias=true, init=:default),
        NNlib.relu,
        BGLinear(input_token_s, 1; bias=true, init=:default),
    )

    to_affinity_pred_score = (
        BGLinear(input_token_s, input_token_s; bias=true, init=:default),
        NNlib.relu,
        BGLinear(input_token_s, input_token_s; bias=true, init=:default),
        NNlib.relu,
        BGLinear(input_token_s, 1; bias=true, init=:default),
    )

    to_affinity_logits_binary = BGLinear(1, 1; bias=true, init=:default)

    return AffinityHeadsTransformer(
        affinity_out_mlp,
        to_affinity_pred_value,
        to_affinity_pred_score,
        to_affinity_logits_binary,
    )
end

function _apply_seq(seq, x)
    out = x
    for layer in seq
        if layer isa Function
            out = layer.(out)
        else
            out = layer(out)
        end
    end
    return out
end

# _unsqueeze_local is defined in confidence.jl (same module, included first)

function (ah::AffinityHeadsTransformer)(
    z,
    feats;
    multiplicity::Int=1,
)
    pad_token_mask_ff = repeat_interleave_batch(feats["token_pad_mask"], multiplicity)
    rec_mask_ff = Float32.(repeat_interleave_batch(feats["mol_type"], multiplicity) .== 0)
    rec_mask_ff = rec_mask_ff .* pad_token_mask_ff
    lig_mask_ff = Float32.(repeat_interleave_batch(feats["affinity_token_mask"], multiplicity) .> 0)
    lig_mask_ff = lig_mask_ff .* pad_token_mask_ff

    cross_pair_mask_ff = _unsqueeze_local(lig_mask_ff, 2) .* _unsqueeze_local(rec_mask_ff, 1) .+ _unsqueeze_local(rec_mask_ff, 2) .* _unsqueeze_local(lig_mask_ff, 1) .+ _unsqueeze_local(lig_mask_ff, 2) .* _unsqueeze_local(lig_mask_ff, 1)
    eye_mask = reshape(1f0 .- Matrix{Float32}(I, size(lig_mask_ff, 1), size(lig_mask_ff, 1)), size(lig_mask_ff, 1), size(lig_mask_ff, 1), 1)
    cross_pair_mask_ff = cross_pair_mask_ff .* eye_mask

    cross_pair_mask_ff = reshape(cross_pair_mask_ff, 1, size(cross_pair_mask_ff, 1), size(cross_pair_mask_ff, 2), size(cross_pair_mask_ff, 3))

    g = sum(z .* cross_pair_mask_ff; dims=(2, 3))
    denom = sum(cross_pair_mask_ff; dims=(2, 3)) .+ 1f-7
    g = dropdims(g ./ denom; dims=(2, 3))

    g = _apply_seq(ah.affinity_out_mlp, g)

    affinity_pred_value = _apply_seq(ah.to_affinity_pred_value, g)
    affinity_pred_score = _apply_seq(ah.to_affinity_pred_score, g)
    affinity_logits_binary = ah.to_affinity_logits_binary(affinity_pred_score)

    out_dict = Dict{String,Any}(
        "affinity_pred_value" => affinity_pred_value,
        "affinity_logits_binary" => affinity_logits_binary,
    )
    return out_dict
end
