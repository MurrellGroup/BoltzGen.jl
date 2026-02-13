using Onion
using NNlib

const BGLayerNorm = Onion.BGLayerNorm

function one_hot(indices::AbstractArray{<:Integer}, num_classes::Int)
    classes_cpu = reshape(collect(0:num_classes-1), ntuple(_ -> 1, ndims(indices))..., num_classes)
    # Transfer classes to same device as indices for GPU compatibility
    classes = copyto!(similar(indices, eltype(classes_cpu), size(classes_cpu)), classes_cpu)
    return Float32.(indices .== classes)
end

@concrete struct FourierEmbedding <: Onion.Layer
    weight
    bias
end

@layer FourierEmbedding

function FourierEmbedding(dim::Int)
    weight = randn(Float32, dim, 1)
    bias = randn(Float32, dim)
    return FourierEmbedding(weight, bias)
end

function (f::FourierEmbedding)(times)
    # times: (B)
    t = reshape(Float32.(times), 1, :)
    rand_proj = f.weight * t .+ reshape(f.bias, :, 1)
    return cos.(2f0 * Float32(pi) .* rand_proj)
end

@concrete struct RelativePositionEncoder <: Onion.Layer
    r_max::Int
    s_max::Int
    linear_layer
end

@layer RelativePositionEncoder

function RelativePositionEncoder(token_z::Int; r_max::Int=32, s_max::Int=2)
    in_dim = 4 * (r_max + 1) + 2 * (s_max + 1) + 1
    linear_layer = LinearNoBias(in_dim, token_z)
    Onion.torch_linear_init!(linear_layer.weight)
    return RelativePositionEncoder(r_max, s_max, linear_layer)
end

function (enc::RelativePositionEncoder)(feats)
    # feats entries expected as (N, B)
    asym = permutedims(feats["feature_asym_id"], (2, 1))
    residue = permutedims(feats["feature_residue_index"], (2, 1))
    entity = permutedims(feats["entity_id"], (2, 1))
    token_index = permutedims(feats["token_index"], (2, 1))
    sym_id = permutedims(feats["sym_id"], (2, 1))
    cyclic = permutedims(feats["cyclic"], (2, 1))

    B, N = size(asym)

    asym_q = reshape(asym, B, N, 1)
    asym_k = reshape(asym, B, 1, N)
    residue_q = reshape(residue, B, N, 1)
    residue_k = reshape(residue, B, 1, N)
    entity_q = reshape(entity, B, N, 1)
    entity_k = reshape(entity, B, 1, N)
    token_q = reshape(token_index, B, N, 1)
    token_k = reshape(token_index, B, 1, N)
    sym_q = reshape(sym_id, B, N, 1)
    sym_k = reshape(sym_id, B, 1, N)

    b_same_chain = asym_q .== asym_k
    b_same_residue = residue_q .== residue_k
    b_same_entity = entity_q .== entity_k

    d_residue = residue_q .- residue_k

    if any(cyclic .> 0)
        period = ifelse.(cyclic .> 0, cyclic, 10000)
        period = reshape(period, B, 1, N)
        d_residue = round.(d_residue ./ period) .* (-period) .+ d_residue
    end

    d_residue = clamp.(d_residue .+ enc.r_max, 0, 2 * enc.r_max)
    d_residue = ifelse.(b_same_chain, d_residue, 2 * enc.r_max + 1)

    a_rel_pos = one_hot(Int.(d_residue), 2 * enc.r_max + 2)

    d_token = clamp.(token_q .- token_k .+ enc.r_max, 0, 2 * enc.r_max)
    d_token = ifelse.(b_same_chain .& b_same_residue, d_token, 2 * enc.r_max + 1)
    a_rel_token = one_hot(Int.(d_token), 2 * enc.r_max + 2)

    d_chain = clamp.(sym_q .- sym_k .+ enc.s_max, 0, 2 * enc.s_max)
    d_chain = ifelse.(.!b_same_entity, 2 * enc.s_max + 1, d_chain)
    a_rel_chain = one_hot(Int.(d_chain), 2 * enc.s_max + 2)

    feats_cat = cat(
        Float32.(a_rel_pos),
        Float32.(a_rel_token),
        reshape(Float32.(b_same_entity), B, N, N, 1),
        Float32.(a_rel_chain);
        dims=4,
    )

    feats_ff = permutedims(feats_cat, (4, 2, 3, 1))
    return enc.linear_layer(feats_ff)
end

@concrete struct SingleConditioning <: Onion.Layer
    sigma_data
    disable_times::Bool
    norm_single
    token_s_to_tfmr_s
    single_embed
    fourier_embed
    norm_fourier
    fourier_to_single
    transitions
end

@layer SingleConditioning

function SingleConditioning(
    sigma_data::Float32;
    tfmr_s::Int=768,
    token_s::Int=384,
    dim_fourier::Int=256,
    num_transitions::Int=2,
    transition_expansion_factor::Int=2,
    eps::Float32=1f-20,
    disable_times::Bool=false,
)
    norm_single = BGLayerNorm(token_s * 2; eps=1f-5)
    token_s_to_tfmr_s = tfmr_s != token_s * 2 ? BGLinear(token_s * 2, tfmr_s; bias=true, init=:default) : nothing
    single_embed = BGLinear(tfmr_s, tfmr_s; bias=true, init=:default)
    if token_s_to_tfmr_s !== nothing
        Onion.torch_linear_init!(token_s_to_tfmr_s.weight, token_s_to_tfmr_s.bias)
    end
    Onion.torch_linear_init!(single_embed.weight, single_embed.bias)

    fourier_embed = disable_times ? nothing : FourierEmbedding(dim_fourier)
    norm_fourier = disable_times ? nothing : BGLayerNorm(dim_fourier; eps=1f-5)
    fourier_to_single = disable_times ? nothing : LinearNoBias(dim_fourier, tfmr_s)
    if fourier_to_single !== nothing
        Onion.torch_linear_init!(fourier_to_single.weight)
    end

    transitions = [Onion.Transition(tfmr_s, transition_expansion_factor * tfmr_s) for _ in 1:num_transitions]

    return SingleConditioning(sigma_data, disable_times, norm_single, token_s_to_tfmr_s, single_embed, fourier_embed, norm_fourier, fourier_to_single, transitions)
end

function (sc::SingleConditioning)(times, s_trunk, s_inputs)
    s = cat(s_trunk, s_inputs; dims=1)
    s = sc.norm_single(s)

    if sc.token_s_to_tfmr_s !== nothing
        s = sc.token_s_to_tfmr_s(s)
    end

    s = sc.single_embed(s)
    normed_fourier = nothing
    if !sc.disable_times
        fourier_embed = sc.fourier_embed(times)
        normed_fourier = sc.norm_fourier(fourier_embed)
        fourier_to_single = sc.fourier_to_single(normed_fourier)
        s = s .+ reshape(fourier_to_single, size(fourier_to_single,1), 1, size(fourier_to_single,2))
    end

    for transition in sc.transitions
        s = transition(s) .+ s
    end

    return s, normed_fourier
end

@concrete struct PairwiseConditioning <: Onion.Layer
    dim_pairwise_init_proj
    transitions
end

@layer PairwiseConditioning

function PairwiseConditioning(token_z::Int, dim_token_rel_pos_feats::Int; num_transitions::Int=2, transition_expansion_factor::Int=2)
    dim_pairwise_init_proj = (BGLayerNorm(token_z + dim_token_rel_pos_feats; eps=1f-5), LinearNoBias(token_z + dim_token_rel_pos_feats, token_z))
    Onion.torch_linear_init!(dim_pairwise_init_proj[2].weight)
    transitions = [Onion.Transition(token_z, transition_expansion_factor * token_z) for _ in 1:num_transitions]
    return PairwiseConditioning(dim_pairwise_init_proj, transitions)
end

function (pc::PairwiseConditioning)(z_trunk, token_rel_pos_feats)
    z = cat(z_trunk, token_rel_pos_feats; dims=1)
    ln, lin = pc.dim_pairwise_init_proj
    z = lin(ln(z))
    for transition in pc.transitions
        z = transition(z) .+ z
    end
    return z
end

@concrete struct CoordinateConditioning <: Onion.Layer
    sigma_data
    single_embed
    disable_times::Bool
    fourier_embed
    norm_fourier
    fourier_to_single
    embed_atom_features
    embed_atompair_ref_coord
    embed_atompair_ref_dist
    embed_atompair_mask
    s_to_c_trans
end

@layer CoordinateConditioning

function CoordinateConditioning(
    sigma_data::Float32,
    atom_s::Int,
    token_s::Int,
    num_heads::Int;
    tfmr_s::Int=768,
    dim_fourier::Int=256,
    atom_feature_dim::Int=132,
    structure_prediction::Bool=true,
    disable_times::Bool=false,
)
    single_embed = LinearNoBias(token_s * 2, tfmr_s)
    Onion.torch_linear_init!(single_embed.weight)
    fourier_embed = disable_times ? nothing : FourierEmbedding(dim_fourier)
    norm_fourier = disable_times ? nothing : BGLayerNorm(dim_fourier; eps=1f-5)
    fourier_to_single = disable_times ? nothing : LinearNoBias(dim_fourier, tfmr_s)

    embed_atom_features = BGLinear(atom_feature_dim, num_heads; bias=true, init=:default)
    embed_atompair_ref_coord = LinearNoBias(3, num_heads)
    embed_atompair_ref_dist = LinearNoBias(1, num_heads)
    embed_atompair_mask = LinearNoBias(1, num_heads)
    Onion.torch_linear_init!(embed_atom_features.weight, embed_atom_features.bias)
    Onion.torch_linear_init!(embed_atompair_ref_coord.weight)
    Onion.torch_linear_init!(embed_atompair_ref_dist.weight)
    Onion.torch_linear_init!(embed_atompair_mask.weight)

    s_to_c_trans = structure_prediction ? (BGLayerNorm(tfmr_s; eps=1f-5), LinearNoBias(tfmr_s, 1)) : nothing
    if s_to_c_trans !== nothing
        Onion.final_init!(s_to_c_trans[2].weight)
    end

    return CoordinateConditioning(sigma_data, single_embed, disable_times, fourier_embed, norm_fourier, fourier_to_single,
        embed_atom_features, embed_atompair_ref_coord, embed_atompair_ref_dist, embed_atompair_mask, s_to_c_trans)
end

function (cc::CoordinateConditioning)(s_trunk, s_inputs, times, feats, atom_coords_noisy)
    s = cat(s_trunk, s_inputs; dims=1)
    s = cc.single_embed(s)

    if !cc.disable_times
        fourier_embed = cc.fourier_embed(times)
        normed_fourier = cc.norm_fourier(fourier_embed)
        fourier_to_single = cc.fourier_to_single(normed_fourier)
        s = s .+ reshape(fourier_to_single, size(fourier_to_single,1), 1, size(fourier_to_single,2))
    end

    atom_mask = feats["atom_pad_mask"]
    atom_ref_pos = feats["ref_pos"]
    atom_uid = feats["ref_space_uid"]

    atom_feats = cat(
        atom_ref_pos,
        reshape(feats["ref_charge"], 1, size(atom_ref_pos, 2), size(atom_ref_pos, 3)),
        feats["ref_element"];
        dims=1,
    )
    c = cc.embed_atom_features(atom_feats)

    B = size(atom_coords_noisy, 3)

    if cc.s_to_c_trans !== nothing
        ln, lin = cc.s_to_c_trans
        s_to_c = lin(ln(s))
        atom_to_token = feats["atom_to_token"]
        s_to_c = NNlib.batched_mul(atom_to_token, permutedims(s_to_c, (2,1,3)))
        s_to_c = permutedims(s_to_c, (2,1,3))
        c = c .+ s_to_c
    end

    atom_mask = repeat(atom_mask, 1, B ÷ size(atom_mask,2))
    atom_uid = repeat(atom_uid, 1, B ÷ size(atom_uid,2))

    # compute pair features
    d = reshape(atom_coords_noisy, 3, size(atom_coords_noisy,2), 1, B) .- reshape(atom_coords_noisy, 3, 1, size(atom_coords_noisy,2), B)
    d_norm = sum(d .* d; dims=1)
    d_norm = 1f0 ./ (1f0 .+ d_norm)

    atom_mask_queries = reshape(atom_mask, 1, size(atom_mask,1), 1, size(atom_mask,2))
    atom_mask_keys = reshape(atom_mask, 1, 1, size(atom_mask,1), size(atom_mask,2))
    atom_uid_queries = reshape(atom_uid, 1, size(atom_uid,1), 1, size(atom_uid,2))
    atom_uid_keys = reshape(atom_uid, 1, 1, size(atom_uid,1), size(atom_uid,2))

    v = (atom_mask_queries .& atom_mask_keys .& (atom_uid_queries .== atom_uid_keys))
    v = Float32.(v)

    p = cc.embed_atompair_ref_coord(d) .* v
    p = p .+ cc.embed_atompair_ref_dist(d_norm) .* v
    p = p .+ cc.embed_atompair_mask(v) .* v
    p = p .+ reshape(c, size(c,1), 1, size(c,2), size(c,3)) .+ reshape(c, size(c,1), size(c,2), 1, size(c,3))

    return sum(p; dims=4)
end

@concrete struct DistanceTokenEncoder <: Onion.Layer
    distance_gaussian_smearing
    distance_token_bias_trans
end

@layer DistanceTokenEncoder

function DistanceTokenEncoder(distance_gaussian_dim::Int, token_z::Int, out_dim::Int)
    distance_gaussian_smearing = GaussianSmearing(start=0.0f0, stop=2.0f0, num_gaussians=distance_gaussian_dim)
    input_dim = distance_gaussian_dim + 1 + token_z
    distance_token_bias_trans = Onion.Transition(input_dim, token_z; out_dim=out_dim)
    return DistanceTokenEncoder(distance_gaussian_smearing, distance_token_bias_trans)
end

function (enc::DistanceTokenEncoder)(relative_position_encoding, feats)
    # relative_position_encoding: (C, N, N, B)
    B = size(relative_position_encoding, 4)
    N = size(relative_position_encoding, 2)

    token_to_bb4_atoms = feats["token_to_bb4_atoms"]
    r = feats["coords"]

    # NNlib.batched_mul expects batch dimension last: (M,K,B) x (K,N,B)
    token_to_bb4 = token_to_bb4_atoms
    r_b = permutedims(r, (2,1,3))

    r_repr = NNlib.batched_mul(token_to_bb4, r_b)
    r_repr = reshape(r_repr, 4, N, 3, B)
    r_repr = permutedims(r_repr, (4, 1, 2, 3))

    d = reshape(r_repr, size(r_repr,1), size(r_repr,2), size(r_repr,3), 1, size(r_repr,4)) .-
        reshape(r_repr, size(r_repr,1), size(r_repr,2), 1, size(r_repr,3), size(r_repr,4))
    d = sqrt.(sum(d .* d; dims=5))
    d = reshape(d, B, 4, N, N, 1)

    distance_gaussian = enc.distance_gaussian_smearing(d)

    rel_pos = permutedims(relative_position_encoding, (4,2,3,1))
    rel_pos = reshape(rel_pos, B, 1, N, N, size(rel_pos,4))
    rel_pos = repeat(rel_pos, 1, 4, 1, 1, 1)

    input_cat = cat(distance_gaussian, d, rel_pos; dims=5)
    input_cat = permutedims(input_cat, (5,3,4,2,1))
    out = enc.distance_token_bias_trans(input_cat)
    out = permutedims(out, (5,2,3,1,4))
    out = reshape(out, B, N, N, size(out,4) * size(out,5))
    return permutedims(out, (4,2,3,1))
end

# windowing helpers
function get_indexing_matrix(K::Int, W::Int, H::Int)
    @assert W % 2 == 0
    @assert H % (W ÷ 2) == 0
    h = H ÷ (W ÷ 2)
    @assert h % 2 == 0

    arange = collect(0:2K-1)
    index = arange' .- arange
    index = index .+ h ÷ 2
    index = clamp.(index, 0, h + 1)
    # take even rows (row-major view would select [:, 0, :])
    index = index[1:2:2K, :]

    onehot = one_hot(Int.(index), h + 2)
    onehot = onehot[:, :, 2:end-1]
    # match python row-major reshape by swapping last two dims before column-major reshape
    onehot = permutedims(onehot, (2,3,1))  # (2K, h, K)
    return reshape(onehot, 2K, h * K)
end

function single_to_keys(single, indexing_matrix, W::Int, H::Int)
    # single: (C, N, B)
    if ndims(single) == 2
        single = reshape(single, 1, size(single,1), size(single,2))
    end

    C, N, B = size(single)
    K = N ÷ W
    # split N dimension to match python's view (row-major)
    x = permutedims(single, (2,3,1))              # (N, B, C)
    x = reshape(x, W ÷ 2, 2K, B, C)               # (i, j, B, C)
    x = permutedims(x, (3,2,1,4))                 # (B, j, i, C)

    h = H ÷ (W ÷ 2)
    index_t = permutedims(indexing_matrix, (2, 1)) # (h*K, 2K)
    # Transfer indexing matrix to same device as input
    index_t = copyto!(similar(single, Float32, size(index_t)), index_t)

    # Replace loop with single matmul: index_t @ x_flat for all (b, i) slices at once
    # x is (B, 2K, W÷2, C) — rearrange so 2K is first dim, flatten the rest
    x_perm = permutedims(x, (2, 1, 3, 4))        # (2K, B, W÷2, C)
    x_flat = reshape(x_perm, 2K, :)               # (2K, B*(W÷2)*C)
    out_flat = index_t * x_flat                    # (h*K, B*(W÷2)*C)
    out = reshape(out_flat, h * K, B, W ÷ 2, C)   # (h*K, B, W÷2, C)
    out = permutedims(out, (2, 1, 3, 4))           # (B, h*K, W÷2, C)

    # match python row-major reshape of (B, k, i, d) -> (B, K, H, D)
    out = permutedims(out, (1,3,2,4))            # (B, i, k, C)
    out = reshape(out, B, H, K, C)               # (B, H, K, C)
    out = permutedims(out, (1,3,2,4))            # (B, K, H, C)
    out = permutedims(out, (4,3,2,1))            # (C, H, K, B)
    return out
end

@concrete struct AtomEncoder <: Onion.Layer
    embed_atom_features
    embed_atompair_ref_pos
    embed_atompair_ref_dist
    embed_atompair_mask
    atoms_per_window_queries::Int
    atoms_per_window_keys::Int
    structure_prediction::Bool
    s_to_c_trans
    z_to_p_trans
    c_to_p_trans_k
    c_to_p_trans_q
    p_mlp
end

@layer AtomEncoder

function AtomEncoder(
    atom_s::Int,
    atom_z::Int,
    token_s::Int,
    token_z::Int,
    atoms_per_window_queries::Int,
    atoms_per_window_keys::Int,
    atom_feature_dim::Int;
    structure_prediction::Bool=true,
)
    embed_atom_features = BGLinear(atom_feature_dim, atom_s; bias=true, init=:default)
    embed_atompair_ref_pos = LinearNoBias(3, atom_z)
    embed_atompair_ref_dist = LinearNoBias(1, atom_z)
    embed_atompair_mask = LinearNoBias(1, atom_z)
    Onion.torch_linear_init!(embed_atom_features.weight, embed_atom_features.bias)
    Onion.torch_linear_init!(embed_atompair_ref_pos.weight)
    Onion.torch_linear_init!(embed_atompair_ref_dist.weight)
    Onion.torch_linear_init!(embed_atompair_mask.weight)

    s_to_c_trans = structure_prediction ? (BGLayerNorm(token_s; eps=1f-5), LinearNoBias(token_s, atom_s)) : nothing
    z_to_p_trans = structure_prediction ? (BGLayerNorm(token_z; eps=1f-5), LinearNoBias(token_z, atom_z)) : nothing
    if s_to_c_trans !== nothing
        Onion.final_init!(s_to_c_trans[2].weight)
    end
    if z_to_p_trans !== nothing
        Onion.final_init!(z_to_p_trans[2].weight)
    end

    c_to_p_trans_k = (NNlib.relu, LinearNoBias(atom_s, atom_z))
    c_to_p_trans_q = (NNlib.relu, LinearNoBias(atom_s, atom_z))
    Onion.final_init!(c_to_p_trans_k[2].weight)
    Onion.final_init!(c_to_p_trans_q[2].weight)

    p_mlp = (
        NNlib.relu,
        LinearNoBias(atom_z, atom_z),
        NNlib.relu,
        LinearNoBias(atom_z, atom_z),
        NNlib.relu,
        LinearNoBias(atom_z, atom_z),
    )
    Onion.torch_linear_init!(p_mlp[2].weight)
    Onion.torch_linear_init!(p_mlp[4].weight)
    Onion.final_init!(p_mlp[6].weight)

    return AtomEncoder(embed_atom_features, embed_atompair_ref_pos, embed_atompair_ref_dist, embed_atompair_mask,
        atoms_per_window_queries, atoms_per_window_keys, structure_prediction, s_to_c_trans, z_to_p_trans, c_to_p_trans_k, c_to_p_trans_q, p_mlp)
end

function (ae::AtomEncoder)(feats; s_trunk=nothing, z=nothing)
    atom_mask = feats["atom_pad_mask"] .> 0.5
    atom_ref_pos = feats["ref_pos"]
    atom_uid = feats["ref_space_uid"]

    atom_feats = cat(
        atom_ref_pos,
        reshape(feats["ref_charge"], 1, size(atom_ref_pos,2), size(atom_ref_pos,3)),
        feats["ref_element"],
        reshape(feats["ref_atom_name_chars"], size(feats["ref_atom_name_chars"],1) * size(feats["ref_atom_name_chars"],2), size(atom_ref_pos,2), size(atom_ref_pos,3));
        dims=1,
    )

    c = ae.embed_atom_features(atom_feats)
    q = c

    W, H = ae.atoms_per_window_queries, ae.atoms_per_window_keys
    N = size(c, 2)
    B = size(c, 3)
    K = N ÷ W
    indexing_matrix = get_indexing_matrix(K, W, H)

    to_keys = x -> single_to_keys(x, indexing_matrix, W, H)

    atom_ref_pos_queries = reshape(atom_ref_pos, 3, W, K, B)
    atom_ref_pos_keys = to_keys(atom_ref_pos)

    # reshape for pairwise: (3, W, 1, K, B) and (3, 1, H, K, B)
    d = reshape(atom_ref_pos_keys, 3, 1, H, K, B) .- reshape(atom_ref_pos_queries, 3, W, 1, K, B)
    d_norm = sum(d .* d; dims=1)
    d_norm = 1f0 ./ (1f0 .+ d_norm)

    atom_mask_queries = reshape(atom_mask, W, K, B)
    atom_mask_keys = reshape(to_keys(Float32.(atom_mask)), H, K, B) .> 0.5
    atom_uid_queries = reshape(atom_uid, W, K, B)
    atom_uid_keys = reshape(to_keys(Float32.(atom_uid)), H, K, B)
    atom_uid_keys = round.(Int, atom_uid_keys)

    v = (reshape(atom_mask_queries, 1, W, 1, K, B) .& reshape(atom_mask_keys, 1, 1, H, K, B) .& (reshape(atom_uid_queries, 1, W, 1, K, B) .== reshape(atom_uid_keys, 1, 1, H, K, B)))
    v = Float32.(v)

    p = ae.embed_atompair_ref_pos(d) .* v
    p = p .+ ae.embed_atompair_ref_dist(d_norm) .* v
    p = p .+ ae.embed_atompair_mask(v) .* v

    if ae.structure_prediction && s_trunk !== nothing && z !== nothing
        atom_to_token = feats["atom_to_token"]
        ln_s, lin_s = ae.s_to_c_trans
        s_to_c = lin_s(ln_s(s_trunk))
        s_to_c = NNlib.batched_mul(atom_to_token, permutedims(s_to_c, (2,1,3)))
        s_to_c = permutedims(s_to_c, (2,1,3))
        c = c .+ s_to_c

        atom_to_token_queries = reshape(atom_to_token, W, K, size(atom_to_token,2), size(atom_to_token,3))
        atom_to_token_keys = to_keys(permutedims(atom_to_token, (2,1,3)))
        ln_z, lin_z = ae.z_to_p_trans
        z_to_p = lin_z(ln_z(z))
        z_bp = permutedims(z_to_p, (4,2,3,1))                # (B, Nt, Nt, D)
        q_map = permutedims(atom_to_token_queries, (4,2,1,3)) # (B, K, W, Nt)
        k_map = permutedims(atom_to_token_keys, (4,3,2,1))    # (B, K, H, Nt)
        D_z = size(z_bp, 4)
        Nt = size(z_bp, 2)

        # Vectorized z_proj: z_proj[b,k,w,h,d] = Σ_nq Σ_nk q_map[b,k,w,nq] * z_bp[b,nq,nk,d] * k_map[b,k,h,nk]

        # Step 1: Contract over nq (query tokens), batched over B
        Q_bat = reshape(permutedims(q_map, (2,3,4,1)), K*W, Nt, B)    # (K*W, Nt, B)
        Z_bat = reshape(permutedims(z_bp, (2,3,4,1)), Nt, Nt*D_z, B)  # (Nt, Nt*D_z, B)
        tmp_bat = NNlib.batched_mul(Q_bat, Z_bat)                       # (K*W, Nt*D_z, B)
        tmp_5d = reshape(tmp_bat, K, W, Nt, D_z, B)                    # (K, W, Nt, D_z, B)

        # Step 2: Contract over nk (key tokens), batched over K*B
        tmp_r = reshape(permutedims(tmp_5d, (2,4,3,1,5)), W*D_z, Nt, K*B)  # (W*D_z, Nt, K*B)
        Kt = reshape(permutedims(k_map, (4,3,2,1)), Nt, H, K*B)            # (Nt, H, K*B)
        result_bat = NNlib.batched_mul(tmp_r, Kt)                            # (W*D_z, H, K*B)
        result = reshape(result_bat, W, D_z, H, K, B)                       # (W, D_z, H, K, B)

        # Permute to (D_z, W, H, K, B) to match p's layout
        z_proj = permutedims(result, (2, 1, 3, 4, 5))
        p = p .+ z_proj
    end

    # c_to_p projections
    relu, lin_k = ae.c_to_p_trans_k
    relu, lin_q = ae.c_to_p_trans_q

    c_q = lin_q(relu(reshape(c, size(c,1), W, K, B)))
    c_k = lin_k(relu(to_keys(c)))
    p = p .+ reshape(c_q, size(c_q,1), W, 1, K, B)
    p = p .+ reshape(c_k, size(c_k,1), 1, H, K, B)

    # p_mlp
    relu1, lin1, relu2, lin2, relu3, lin3 = ae.p_mlp
    p = p .+ lin3(relu3(lin2(relu2(lin1(relu1(p))))))

    return q, c, p, to_keys
end

@concrete struct AtomAttentionEncoder <: Onion.Layer
    structure_prediction::Bool
    gaussian_random_3d_encoding_dim::Int
    encoding_3d
    r_to_q_trans
    atom_encoder
    atom_to_token_trans
    atoms_per_window_queries::Int
    atoms_per_window_keys::Int
end

@layer AtomAttentionEncoder

function AtomAttentionEncoder(
    atom_s::Int,
    token_s::Int,
    atoms_per_window_queries::Int,
    atoms_per_window_keys::Int;
    atom_encoder_depth::Int=3,
    atom_encoder_heads::Int=4,
    structure_prediction::Bool=true,
    activation_checkpointing::Bool=false,
    gaussian_random_3d_encoding_dim::Int=0,
    transformer_post_layer_norm::Bool=false,
    tfmr_s::Int=token_s,
    use_qk_norm::Bool=false,
)
    encoding_3d = gaussian_random_3d_encoding_dim > 0 ? GaussianRandom3DEncodings(gaussian_random_3d_encoding_dim) : nothing
    r_input_size = 3 + gaussian_random_3d_encoding_dim
    r_to_q_trans = structure_prediction ? LinearNoBias(r_input_size, atom_s) : nothing
    if r_to_q_trans !== nothing
        Onion.final_init!(r_to_q_trans.weight)
    end

    atom_encoder = AtomTransformer(
        attn_window_queries=atoms_per_window_queries,
        attn_window_keys=atoms_per_window_keys,
        depth=atom_encoder_depth,
        heads=atom_encoder_heads,
        dim=atom_s,
        dim_single_cond=atom_s,
        activation_checkpointing=activation_checkpointing,
        post_layer_norm=transformer_post_layer_norm,
        use_qk_norm=use_qk_norm,
    )

    atom_to_token_trans = (LinearNoBias(atom_s, structure_prediction ? tfmr_s : token_s), NNlib.relu)
    Onion.torch_linear_init!(atom_to_token_trans[1].weight)

    return AtomAttentionEncoder(structure_prediction, gaussian_random_3d_encoding_dim, encoding_3d, r_to_q_trans, atom_encoder, atom_to_token_trans, atoms_per_window_queries, atoms_per_window_keys)
end

function (ae::AtomAttentionEncoder)(; feats, q, c, atom_enc_bias, to_keys, r=nothing, multiplicity::Int=1)
    atom_mask = feats["atom_pad_mask"]

    if ae.structure_prediction
        q = repeat_interleave_batch(q, multiplicity)
        r_input = r
        if ae.gaussian_random_3d_encoding_dim > 0
            r_input = cat(r_input, ae.encoding_3d(r); dims=1)
        end
        r_to_q = ae.r_to_q_trans(r_input)
        q = q .+ r_to_q
    end

    c = repeat_interleave_batch(c, multiplicity)
    atom_mask = repeat_interleave_batch(atom_mask, multiplicity)

    q = ae.atom_encoder(q, c; bias=atom_enc_bias, mask=atom_mask, multiplicity=multiplicity, to_keys=to_keys)

    q_to_a = ae.atom_to_token_trans[2](ae.atom_to_token_trans[1](q))
    atom_to_token = feats["atom_to_token"]
    atom_to_token = repeat_interleave_batch(atom_to_token, multiplicity)
    atom_to_token_mean = atom_to_token ./ (sum(atom_to_token; dims=1) .+ 1f-6)

    a = NNlib.batched_mul(permutedims(atom_to_token_mean, (2,1,3)), permutedims(q_to_a, (2,1,3)))
    a = permutedims(a, (2,1,3))

    return a, q, c, to_keys
end

@concrete struct AtomAttentionDecoder <: Onion.Layer
    predict_res_type::Bool
    a_to_q_trans
    atom_decoder
    atom_feat_to_atom_pos_update
    res_type_predictor
end

@layer AtomAttentionDecoder

function AtomAttentionDecoder(
    atom_s::Int,
    tfmr_s::Int,
    attn_window_queries::Int,
    attn_window_keys::Int;
    atom_decoder_depth::Int=3,
    atom_decoder_heads::Int=4,
    activation_checkpointing::Bool=false,
    transformer_post_layer_norm::Bool=false,
    predict_res_type::Bool=false,
    use_qk_norm::Bool=false,
)
    a_to_q_trans = LinearNoBias(tfmr_s, atom_s)
    Onion.final_init!(a_to_q_trans.weight)

    atom_decoder = AtomTransformer(
        attn_window_queries=attn_window_queries,
        attn_window_keys=attn_window_keys,
        depth=atom_decoder_depth,
        heads=atom_decoder_heads,
        dim=atom_s,
        dim_single_cond=atom_s,
        activation_checkpointing=activation_checkpointing,
        post_layer_norm=transformer_post_layer_norm,
        use_qk_norm=use_qk_norm,
    )

    atom_feat_to_atom_pos_update = (BGLayerNorm(atom_s; eps=1f-5), LinearNoBias(atom_s, 3))
    Onion.final_init!(atom_feat_to_atom_pos_update[2].weight)

    res_type_predictor = predict_res_type ? BGLinear(atom_s, num_tokens; bias=true, init=:default) : nothing
    if res_type_predictor !== nothing
        Onion.torch_linear_init!(res_type_predictor.weight, res_type_predictor.bias)
    end

    return AtomAttentionDecoder(predict_res_type, a_to_q_trans, atom_decoder, atom_feat_to_atom_pos_update, res_type_predictor)
end

function (ad::AtomAttentionDecoder)(; a, q, c, atom_dec_bias, feats, to_keys, multiplicity::Int=1)
    atom_to_token = feats["atom_to_token"]
    atom_to_token = repeat_interleave_batch(atom_to_token, multiplicity)

    a_to_q = ad.a_to_q_trans(a)
    a_to_q = NNlib.batched_mul(atom_to_token, permutedims(a_to_q, (2,1,3)))
    a_to_q = permutedims(a_to_q, (2,1,3))

    q = q .+ a_to_q
    atom_mask = feats["atom_pad_mask"]
    atom_mask = repeat_interleave_batch(atom_mask, multiplicity)

    q = ad.atom_decoder(q, c; bias=atom_dec_bias, mask=atom_mask, multiplicity=multiplicity, to_keys=to_keys)

    res_type = nothing
    if ad.predict_res_type
        idx = mapslices(argmax, feats["atom_to_token"], dims=1)
        idx = repeat_interleave_batch(idx, multiplicity)
        mask = repeat_interleave_batch(feats["atom_pad_mask"], multiplicity)
        src = q .* reshape(mask, 1, size(mask,1), size(mask,2))
        idx_expanded = reshape(idx, 1, size(idx,1), size(idx,2))
        s_feat = Onion.zeros_like(q, Float32, size(q,1), size(feats["res_type"],2), size(q,3))
        for b in 1:size(q,3)
            for m in 1:size(q,2)
                n = idx_expanded[1, m, b]
                if n >= 1 && n <= size(s_feat,2)
                    @inbounds s_feat[:, n, b] .+= src[:, m, b]
                end
            end
        end
        res_type = ad.res_type_predictor(s_feat)
    end

    ln, lin = ad.atom_feat_to_atom_pos_update
    r_update = lin(ln(q))
    return r_update, res_type
end
