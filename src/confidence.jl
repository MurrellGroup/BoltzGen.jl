using Onion
using NNlib

const BGLayerNorm = Onion.BGLayerNorm

@concrete struct ConfidenceModule <: Onion.Layer
    token_z::Int
    no_update_s::Bool
    max_num_atoms_per_token::Int
    boundaries
    dist_bin_pairwise_embed
    s_to_z
    s_to_z_transpose
    add_s_to_z_prod::Bool
    s_to_z_prod_in1
    s_to_z_prod_in2
    s_to_z_prod_out
    s_inputs_norm
    s_norm
    z_norm
    add_s_input_to_s::Bool
    s_input_to_s
    add_z_input_to_z
    rel_pos
    token_bonds
    bond_type_feature::Bool
    token_bonds_type
    contact_conditioning
    pairformer_stack
    return_latent_feats::Bool
    confidence_heads
end

@layer ConfidenceModule

function ConfidenceModule(
    token_s::Int,
    token_z::Int;
    pairformer_args::Dict=Dict(),
    num_dist_bins::Int=64,
    token_level_confidence::Bool=true,
    max_dist::Real=22,
    add_s_to_z_prod::Bool=false,
    confidence_args::Dict=Dict(),
    return_latent_feats::Bool=false,
    conditioning_cutoff_min=nothing,
    conditioning_cutoff_max=nothing,
    bond_type_feature::Bool=false,
    add_z_input_to_z=nothing,
    add_s_input_to_s::Bool=false,
    no_update_s::Bool=false,
)
    max_num_atoms_per_token = 23
    boundaries = collect(range(2f0, Float32(max_dist); length=num_dist_bins - 1))
    dist_bin_pairwise_embed = BGEmbedding(num_dist_bins, token_z; init=:zeros)

    s_to_z = LinearNoBias(token_s, token_z)
    s_to_z_transpose = LinearNoBias(token_s, token_z)
    Onion.gating_init!(s_to_z.weight)
    Onion.gating_init!(s_to_z_transpose.weight)

    if add_s_to_z_prod
        s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        s_to_z_prod_in2 = LinearNoBias(token_s, token_z)
        s_to_z_prod_out = LinearNoBias(token_z, token_z)
        Onion.gating_init!(s_to_z_prod_out.weight)
    else
        s_to_z_prod_in1 = nothing
        s_to_z_prod_in2 = nothing
        s_to_z_prod_out = nothing
    end

    s_inputs_norm = BGLayerNorm(token_s; eps=1f-5)
    s_norm = no_update_s ? nothing : BGLayerNorm(token_s; eps=1f-5)
    z_norm = BGLayerNorm(token_z; eps=1f-5)

    if add_s_input_to_s
        s_input_to_s = LinearNoBias(token_s, token_s)
        Onion.gating_init!(s_input_to_s.weight)
    else
        s_input_to_s = nothing
    end

    if add_z_input_to_z === true
        rel_pos = RelativePositionEncoder(token_z)
        token_bonds = LinearNoBias(1, token_z)
        Onion.torch_linear_init!(token_bonds.weight)
        token_bonds_type = bond_type_feature ? BGEmbedding(length(bond_types) + 1, token_z; init=:torch) : nothing
        contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=Float32(default(conditioning_cutoff_min, 4.0)),
            cutoff_max=Float32(default(conditioning_cutoff_max, 20.0)),
        )
    else
        rel_pos = nothing
        token_bonds = nothing
        token_bonds_type = nothing
        contact_conditioning = nothing
    end

    pairformer_stack = Onion.PairformerModule(
        token_s,
        token_z,
        get(pairformer_args, :num_blocks, 1);
        num_heads=get(pairformer_args, :num_heads, 16),
        dropout=Float32(get(pairformer_args, :dropout, 0.25)),
        pairwise_head_width=get(pairformer_args, :pairwise_head_width, 32),
        pairwise_num_heads=get(pairformer_args, :pairwise_num_heads, 4),
        post_layer_norm=get(pairformer_args, :post_layer_norm, false),
        activation_checkpointing=get(pairformer_args, :activation_checkpointing, false),
    )

    confidence_heads = ConfidenceHeads(
        token_s,
        token_z;
        token_level_confidence=token_level_confidence,
        confidence_args...,
    )

    return ConfidenceModule(
        token_z,
        no_update_s,
        max_num_atoms_per_token,
        boundaries,
        dist_bin_pairwise_embed,
        s_to_z,
        s_to_z_transpose,
        add_s_to_z_prod,
        s_to_z_prod_in1,
        s_to_z_prod_in2,
        s_to_z_prod_out,
        s_inputs_norm,
        s_norm,
        z_norm,
        add_s_input_to_s,
        s_input_to_s,
        add_z_input_to_z,
        rel_pos,
        token_bonds,
        bond_type_feature,
        token_bonds_type,
        contact_conditioning,
        pairformer_stack,
        return_latent_feats,
        confidence_heads,
    )
end

function _maybe_repeat_interleave(x, m::Int)
    m == 1 && return x
    return repeat_interleave_batch(x, m)
end

function _to_py_s(s)
    return permutedims(s, (3, 2, 1))
end

function _to_py_z(z)
    return permutedims(z, (4, 2, 3, 1))
end

function _to_py_logits4(x)
    return permutedims(x, (4, 2, 3, 1))
end

function _to_py_logits3(x)
    return permutedims(x, (3, 2, 1))
end

function _to_ff_logits4(x)
    return permutedims(x, (4, 2, 3, 1))
end

function _to_ff_logits3(x)
    return permutedims(x, (3, 2, 1))
end

function _unsqueeze_local(x, dim::Int)
    sz = size(x)
    new_sz = ntuple(i -> i < dim ? sz[i] : (i == dim ? 1 : sz[i-1]), ndims(x) + 1)
    return reshape(x, new_sz)
end

function (cm::ConfidenceModule)(
    s_inputs,
    s,
    z,
    x_pred,
    feats,
    pred_distogram_logits;
    multiplicity::Int=1,
    s_diffusion=nothing,
    run_sequentially::Bool=false,
    use_kernels::Bool=false,
)
    if run_sequentially && multiplicity > 1
        size(z, 4) == 1 || error("Sequential confidence only supports batch size 1")
        out_dicts = Vector{Dict{String,Any}}(undef, multiplicity)
        for sample_idx in 1:multiplicity
            x_pred_i = x_pred[:, :, sample_idx:sample_idx]
            out_dicts[sample_idx] = cm(
                s_inputs,
                s,
                z,
                x_pred_i,
                feats,
                pred_distogram_logits;
                multiplicity=1,
                s_diffusion=nothing,
                run_sequentially=false,
                use_kernels=use_kernels,
            )
        end
        out = Dict{String,Any}()
        for key in keys(out_dicts[1])
            if key != "pair_chains_iptm"
                vals = [out_dicts[i][key] for i in 1:length(out_dicts)]
                if vals[1] isa AbstractArray
                    out[key] = cat(vals...; dims=ndims(vals[1]))
                else
                    out[key] = vals
                end
            else
                pair_chains_iptm = Dict{Int,Dict{Int,Any}}()
                first_dict = out_dicts[1][key]
                for chain_idx1 in keys(first_dict)
                    chains_iptm = Dict{Int,Any}()
                    for chain_idx2 in keys(first_dict[chain_idx1])
                        vals = [out_dicts[i][key][chain_idx1][chain_idx2] for i in 1:length(out_dicts)]
                        chains_iptm[chain_idx2] = cat(vals...; dims=ndims(vals[1]))
                    end
                    pair_chains_iptm[chain_idx1] = chains_iptm
                end
                out[key] = pair_chains_iptm
            end
        end
        return out
    end

    s_inputs = cm.s_inputs_norm(s_inputs)
    if !cm.no_update_s && cm.s_norm !== nothing
        s = cm.s_norm(s)
    end

    if cm.add_s_input_to_s && cm.s_input_to_s !== nothing
        s = s .+ cm.s_input_to_s(s_inputs)
    end

    z = cm.z_norm(z)

    s = _maybe_repeat_interleave(s, multiplicity)

    token_z = cm.token_z
    N = size(s_inputs, 2)
    B = size(s_inputs, 3)
    z = z .+ reshape(cm.s_to_z(s_inputs), token_z, N, 1, B) .+ reshape(cm.s_to_z_transpose(s_inputs), token_z, 1, N, B)

    if cm.add_s_to_z_prod && cm.s_to_z_prod_out !== nothing
        prod = reshape(cm.s_to_z_prod_in1(s_inputs), token_z, N, 1, B) .* reshape(cm.s_to_z_prod_in2(s_inputs), token_z, 1, N, B)
        z = z .+ cm.s_to_z_prod_out(prod)
    end

    z = _maybe_repeat_interleave(z, multiplicity)
    s_inputs = _maybe_repeat_interleave(s_inputs, multiplicity)

    token_to_rep_atom_ff = repeat_interleave_batch(feats["token_to_rep_atom"], multiplicity)

    x_pred_ff = x_pred
    if ndims(x_pred_ff) == 4
        # assume (3, Natoms, B, mult)
        x_pred_ff = reshape(x_pred_ff, size(x_pred_ff, 1), size(x_pred_ff, 2), size(x_pred_ff, 3) * size(x_pred_ff, 4))
    end

    x_pred_batched = permutedims(x_pred_ff, (2, 1, 3))
    x_pred_repr = NNlib.batched_mul(token_to_rep_atom_ff, x_pred_batched)
    x_pred_repr_ff = permutedims(x_pred_repr, (2, 1, 3))

    d = pairwise_distance_batch(permutedims(x_pred_repr_ff, (3, 2, 1)))

    dist_idx = sum(reshape(d, size(d, 1), size(d, 2), size(d, 3), 1) .> reshape(cm.boundaries, 1, 1, 1, :); dims=4)
    dist_idx = Int.(dropdims(dist_idx; dims=4))
    dist_idx_ff = permutedims(dist_idx, (2, 3, 1))
    distogram = cm.dist_bin_pairwise_embed(dist_idx_ff)
    z = z .+ distogram

    mask = _maybe_repeat_interleave(feats["token_pad_mask"], multiplicity)
    pair_mask = reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2))

    s_t, z_t = cm.pairformer_stack(s, z, mask, pair_mask; use_kernels=use_kernels)
    s = s_t
    z = z_t

    out_dict = Dict{String,Any}()
    if cm.return_latent_feats
        out_dict["s_conf"] = s
        out_dict["z_conf"] = z
    end

    out_heads = cm.confidence_heads(
        s,
        z,
        x_pred_ff,
        d,
        feats,
        pred_distogram_logits;
        multiplicity=multiplicity,
    )
    for (k, v) in out_heads
        out_dict[k] = v
    end

    return out_dict
end

@concrete struct ConfidenceHeads <: Onion.Layer
    max_num_atoms_per_token::Int
    token_level_confidence::Bool
    use_separate_heads::Bool
    to_pae_logits
    to_pae_intra_logits
    to_pae_inter_logits
    to_pde_logits
    to_pde_intra_logits
    to_pde_inter_logits
    to_plddt_logits
    to_resolved_logits
end

@layer ConfidenceHeads

function ConfidenceHeads(
    token_s::Int,
    token_z::Int;
    num_plddt_bins::Int=50,
    num_pde_bins::Int=64,
    num_pae_bins::Int=64,
    token_level_confidence::Bool=true,
    use_separate_heads::Bool=false,
)
    if use_separate_heads
        to_pae_intra_logits = LinearNoBias(token_z, num_pae_bins)
        to_pae_inter_logits = LinearNoBias(token_z, num_pae_bins)
        to_pae_logits = nothing
    else
        to_pae_logits = LinearNoBias(token_z, num_pae_bins)
        to_pae_intra_logits = nothing
        to_pae_inter_logits = nothing
    end

    if use_separate_heads
        to_pde_intra_logits = LinearNoBias(token_z, num_pde_bins)
        to_pde_inter_logits = LinearNoBias(token_z, num_pde_bins)
        to_pde_logits = nothing
    else
        to_pde_logits = LinearNoBias(token_z, num_pde_bins)
        to_pde_intra_logits = nothing
        to_pde_inter_logits = nothing
    end

    if token_level_confidence
        to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
        to_resolved_logits = LinearNoBias(token_s, 2)
    else
        to_plddt_logits = LinearNoBias(token_s, num_plddt_bins * 23)
        to_resolved_logits = LinearNoBias(token_s, 2 * 23)
    end

    return ConfidenceHeads(
        23,
        token_level_confidence,
        use_separate_heads,
        to_pae_logits,
        to_pae_intra_logits,
        to_pae_inter_logits,
        to_pde_logits,
        to_pde_intra_logits,
        to_pde_inter_logits,
        to_plddt_logits,
        to_resolved_logits,
    )
end

function (ch::ConfidenceHeads)(
    s,
    z,
    x_pred,
    d,
    feats,
    pred_distogram_logits;
    multiplicity::Int=1,
)
    Bm = size(s, 3)
    N = size(s, 2)

    s_py = _to_py_s(s)
    z_py = _to_py_z(z)

    token_pad_mask_ff = repeat_interleave_batch(feats["token_pad_mask"], multiplicity)
    token_pad_mask_py = permutedims(token_pad_mask_ff, (2, 1))
    asym_id_ff = repeat_interleave_batch(feats["asym_id"], multiplicity)
    asym_id_py = permutedims(asym_id_ff, (2, 1))
    token_type_ff = repeat_interleave_batch(feats["mol_type"], multiplicity)
    token_type_py = permutedims(token_type_ff, (2, 1))
    design_mask_ff = repeat_interleave_batch(feats["design_mask"], multiplicity)
    design_mask_py = permutedims(design_mask_ff, (2, 1))
    chain_design_mask_ff = repeat_interleave_batch(feats["chain_design_mask"], multiplicity)
    chain_design_mask_py = permutedims(chain_design_mask_ff, (2, 1))

    if ch.use_separate_heads
        is_same_chain_ff = _unsqueeze_local(asym_id_ff, 2) .== _unsqueeze_local(asym_id_ff, 1)
        is_different_chain_ff = .!is_same_chain_ff

        pae_intra_logits = ch.to_pae_intra_logits(z)
        pae_intra_logits = pae_intra_logits .* reshape(Float32.(is_same_chain_ff), 1, size(is_same_chain_ff, 1), size(is_same_chain_ff, 2), size(is_same_chain_ff, 3))

        pae_inter_logits = ch.to_pae_inter_logits(z)
        pae_inter_logits = pae_inter_logits .* reshape(Float32.(is_different_chain_ff), 1, size(is_different_chain_ff, 1), size(is_different_chain_ff, 2), size(is_different_chain_ff, 3))

        pae_logits = pae_inter_logits .+ pae_intra_logits
        pae_logits_py = _to_py_logits4(pae_logits)
    else
        pae_logits = ch.to_pae_logits(z)
        pae_logits_py = _to_py_logits4(pae_logits)
    end

    if ch.use_separate_heads
        z_sym = z .+ permutedims(z, (1, 3, 2, 4))
        pde_intra_logits = ch.to_pde_intra_logits(z_sym)
        pde_intra_logits = pde_intra_logits .* reshape(Float32.(is_same_chain_ff), 1, size(is_same_chain_ff, 1), size(is_same_chain_ff, 2), size(is_same_chain_ff, 3))
        pde_inter_logits = ch.to_pde_inter_logits(z_sym)
        pde_inter_logits = pde_inter_logits .* reshape(Float32.(is_different_chain_ff), 1, size(is_different_chain_ff, 1), size(is_different_chain_ff, 2), size(is_different_chain_ff, 3))
        pde_logits = pde_inter_logits .+ pde_intra_logits
        pde_logits_py = _to_py_logits4(pde_logits)
    else
        pde_logits = ch.to_pde_logits(z .+ permutedims(z, (1, 3, 2, 4)))
        pde_logits_py = _to_py_logits4(pde_logits)
    end

    resolved_logits = ch.to_resolved_logits(s)
    plddt_logits = ch.to_plddt_logits(s)

    ligand_weight = 20f0
    non_interface_weight = 1f0
    interface_weight = 10f0

    is_ligand_token = Float32.(token_type_py .== chain_type_ids["NONPOLYMER"])
    is_protein_token = Float32.(token_type_py .== chain_type_ids["PROTEIN"])

    if ch.token_level_confidence
        plddt_logits_py = _to_py_logits3(plddt_logits)
        plddt = Onion.compute_aggregated_metric(plddt_logits_py)
        complex_plddt = sum(plddt .* token_pad_mask_py; dims=2) ./ sum(token_pad_mask_py; dims=2)
        complex_plddt = dropdims(complex_plddt; dims=2)

        is_contact = Float32.(d .< 8f0)
        is_different_chain = Float32.(_unsqueeze_local(asym_id_py, 3) .!= _unsqueeze_local(asym_id_py, 2))

        token_interface_mask = maximum(is_contact .* is_different_chain .* reshape(1f0 .- is_ligand_token, size(is_ligand_token, 1), size(is_ligand_token, 2), 1); dims=3)
        token_interface_mask = dropdims(token_interface_mask; dims=3)
        token_non_interface_mask = (1f0 .- token_interface_mask) .* (1f0 .- is_ligand_token)

        iplddt_weight = is_ligand_token .* ligand_weight .+ token_interface_mask .* interface_weight .+ token_non_interface_mask .* non_interface_weight
        complex_iplddt = sum(plddt .* token_pad_mask_py .* iplddt_weight; dims=2) ./ sum(token_pad_mask_py .* iplddt_weight; dims=2)
        complex_iplddt = dropdims(complex_iplddt; dims=2)

        plddt_ff = permutedims(plddt, (2, 1))
        resolved_logits_ff = resolved_logits
        plddt_logits_ff = plddt_logits
        atom_pad_mask = nothing
    else
        B, Nt, _ = size(resolved_logits)
        resolved_logits_py = _to_py_logits3(resolved_logits)
        plddt_logits_py = _to_py_logits3(plddt_logits)

        resolved_logits_py = reshape(resolved_logits_py, B, Nt, ch.max_num_atoms_per_token, 2)
        plddt_logits_py = reshape(plddt_logits_py, B, Nt, ch.max_num_atoms_per_token, :)

        atom_to_token_py = permutedims(feats["atom_to_token"], (3, 1, 2))
        atoms_per_token = dropdims(sum(atom_to_token_py; dims=2); dims=2)
        arange_max = reshape(collect(0:ch.max_num_atoms_per_token-1), 1, 1, ch.max_num_atoms_per_token)
        max_num_atoms_mask = _unsqueeze_local(atoms_per_token, 3) .> arange_max

        resolved_logits_flat = reshape(resolved_logits_py, B, :, 2)
        plddt_logits_flat = reshape(plddt_logits_py, B, :, size(plddt_logits_py, 4))
        mask_flat = reshape(max_num_atoms_mask, B, :)

        resolved_list = Vector{Array{Float32,2}}(undef, B)
        plddt_list = Vector{Array{Float32,2}}(undef, B)
        for b in 1:B
            resolved_list[b] = resolved_logits_flat[b, mask_flat[b, :], :]
            plddt_list[b] = plddt_logits_flat[b, mask_flat[b, :], :]
        end

        max_atoms = size(feats["atom_pad_mask"], 1)
        resolved_logits_py = zeros(Float32, B, max_atoms, 2)
        plddt_logits_py = zeros(Float32, B, max_atoms, size(plddt_list[1], 2))
        for b in 1:B
            n = size(resolved_list[b], 1)
            resolved_logits_py[b, 1:n, :] .= resolved_list[b]
            plddt_logits_py[b, 1:n, :] .= plddt_list[b]
        end

        atom_pad_mask = permutedims(repeat_interleave_batch(feats["atom_pad_mask"], multiplicity), (2, 1))
        plddt = Onion.compute_aggregated_metric(plddt_logits_py)
        complex_plddt = sum(plddt .* atom_pad_mask; dims=2) ./ sum(atom_pad_mask; dims=2)
        complex_plddt = dropdims(complex_plddt; dims=2)

        atom_to_token_b = repeat_interleave_dim(Float32.(atom_to_token_py), multiplicity, 1)
        token_type_b = repeat_interleave_dim(Float32.(token_type_py), multiplicity, 1)
        chain_id_token_b = repeat_interleave_dim(Float32.(asym_id_py), multiplicity, 1)

        token_type_batched = reshape(token_type_b, size(token_type_b, 1), size(token_type_b, 2), 1)
        atom_type = NNlib.batched_mul(atom_to_token_b, token_type_batched)
        atom_type = dropdims(atom_type; dims=3)

        is_ligand_atom = Float32.(atom_type .== chain_type_ids["NONPOLYMER"])

        x_pred_py = permutedims(x_pred, (3, 2, 1))
        d_atom = pairwise_distance_batch(x_pred_py)
        is_contact = Float32.(d_atom .< 8f0)

        chain_id_token_batched = reshape(chain_id_token_b, size(chain_id_token_b, 1), size(chain_id_token_b, 2), 1)
        chain_id_atom = NNlib.batched_mul(atom_to_token_b, chain_id_token_batched)
        chain_id_atom = dropdims(chain_id_atom; dims=3)
        is_different_chain = Float32.(_unsqueeze_local(chain_id_atom, 3) .!= _unsqueeze_local(chain_id_atom, 2))

        atom_interface_mask = maximum(is_contact .* is_different_chain .* reshape(1f0 .- is_ligand_atom, size(is_ligand_atom, 1), size(is_ligand_atom, 2), 1); dims=3)
        atom_interface_mask = dropdims(atom_interface_mask; dims=3)
        atom_non_interface_mask = (1f0 .- atom_interface_mask) .* (1f0 .- is_ligand_atom)

        iplddt_weight = is_ligand_atom .* ligand_weight .+ atom_interface_mask .* interface_weight .+ atom_non_interface_mask .* non_interface_weight
        complex_iplddt = sum(plddt .* atom_pad_mask .* iplddt_weight; dims=2) ./ sum(atom_pad_mask .* iplddt_weight; dims=2)
        complex_iplddt = dropdims(complex_iplddt; dims=2)

        plddt_ff = permutedims(plddt, (2, 1))
        resolved_logits_ff = permutedims(resolved_logits_py, (3, 2, 1))
        plddt_logits_ff = permutedims(plddt_logits_py, (3, 2, 1))
    end

    pde = Onion.compute_aggregated_metric(pde_logits_py; end_value=32f0)
    pae = Onion.compute_aggregated_metric(pae_logits_py; end_value=32f0)

    pred_dist_py = _to_py_logits4(pred_distogram_logits)
    pred_dist_prob = NNlib.softmax(pred_dist_py; dims=4)
    pred_dist_prob = repeat_interleave_dim(pred_dist_prob, multiplicity, 1)

    contacts = zeros(Float32, 1, 1, 1, 64)
    contacts[:, :, :, 1:20] .= 1f0
    prob_contact = sum(pred_dist_prob .* contacts; dims=4)
    prob_contact = dropdims(prob_contact; dims=4)

    eye_mask = reshape(1f0 .- Matrix{Float32}(I, N, N), 1, N, N)
    token_pad_pair_mask = _unsqueeze_local(token_pad_mask_py, 3) .* _unsqueeze_local(token_pad_mask_py, 2) .* eye_mask

    selected_ligand_mask = is_ligand_token .* Float32.(asym_id_py .== 1)

    protein_pair_mask = token_pad_pair_mask .* (_unsqueeze_local(is_protein_token, 3) .* _unsqueeze_local(is_protein_token, 2))
    protein_pair_mask_weighted = protein_pair_mask .* prob_contact
    protein_pae_value_weighted = sum(pae .* protein_pair_mask_weighted; dims=(2, 3)) ./ (sum(protein_pair_mask_weighted; dims=(2, 3)) .+ 1f-5)
    protein_pae_value_weighted = dropdims(protein_pae_value_weighted; dims=(2, 3))
    protein_pae_value_unweighted = sum(pae .* protein_pair_mask; dims=(2, 3)) ./ (sum(protein_pair_mask; dims=(2, 3)) .+ 1f-5)
    protein_pae_value_unweighted = dropdims(protein_pae_value_unweighted; dims=(2, 3))

    ligand_pair_mask = token_pad_pair_mask .* (_unsqueeze_local(selected_ligand_mask, 3) .* _unsqueeze_local(selected_ligand_mask, 2))
    ligand_pair_mask_weighted = ligand_pair_mask .* prob_contact
    ligand_pae_value_weighted = sum(pae .* ligand_pair_mask_weighted; dims=(2, 3)) ./ (sum(ligand_pair_mask_weighted; dims=(2, 3)) .+ 1f-5)
    ligand_pae_value_weighted = dropdims(ligand_pae_value_weighted; dims=(2, 3))
    ligand_pae_value_unweighted = sum(pae .* ligand_pair_mask; dims=(2, 3)) ./ (sum(ligand_pair_mask; dims=(2, 3)) .+ 1f-5)
    ligand_pae_value_unweighted = dropdims(ligand_pae_value_unweighted; dims=(2, 3))

    interface_protein_mask = maximum(Float32.(d .< 8f0) .* _unsqueeze_local(is_protein_token, 3) .* _unsqueeze_local(selected_ligand_mask, 2); dims=3)
    interface_protein_mask = dropdims(interface_protein_mask; dims=3)
    protein_ligand_interface_pair_mask = token_pad_pair_mask .* _unsqueeze_local(interface_protein_mask, 3) .* _unsqueeze_local(selected_ligand_mask, 2) .* Float32.(d .< 8f0)
    protein_ligand_interface_pair_mask_weighted = protein_ligand_interface_pair_mask .* prob_contact
    interface_pae_value_weighted = sum(pae .* protein_ligand_interface_pair_mask_weighted; dims=(2, 3)) ./ (sum(protein_ligand_interface_pair_mask_weighted; dims=(2, 3)) .+ 1f-5)
    interface_pae_value_weighted = dropdims(interface_pae_value_weighted; dims=(2, 3))
    interface_pae_value_unweighted = sum(pae .* protein_ligand_interface_pair_mask; dims=(2, 3)) ./ (sum(protein_ligand_interface_pair_mask; dims=(2, 3)) .+ 1f-5)
    interface_pae_value_unweighted = dropdims(interface_pae_value_unweighted; dims=(2, 3))

    token_pair_mask = token_pad_pair_mask .* prob_contact
    complex_pde = sum(pde .* token_pair_mask; dims=(2, 3)) ./ sum(token_pair_mask; dims=(2, 3))
    complex_pde = dropdims(complex_pde; dims=(2, 3))
    token_interface_pair_mask = token_pair_mask .* Float32.(_unsqueeze_local(asym_id_py, 3) .!= _unsqueeze_local(asym_id_py, 2))
    complex_ipde = sum(pde .* token_interface_pair_mask; dims=(2, 3)) ./ (sum(token_interface_pair_mask; dims=(2, 3)) .+ 1f-5)
    complex_ipde = dropdims(complex_ipde; dims=(2, 3))

    is_chain_design_token = if sum(feats["design_mask"]) > 0
        chain_design_mask_py
    else
        design_mask_py
    end
    is_target_token = if sum(feats["design_mask"]) > 0
        1f0 .- chain_design_mask_py
    else
        design_mask_py
    end

    target_designchain_inter_mask = token_pad_pair_mask .* (
        _unsqueeze_local(is_chain_design_token, 3) .* _unsqueeze_local(is_target_token, 2) .+ _unsqueeze_local(is_target_token, 3) .* _unsqueeze_local(is_chain_design_token, 2)
    )

    interaction_pae = sum(pae .* target_designchain_inter_mask; dims=(2, 3)) ./ (sum(target_designchain_inter_mask; dims=(2, 3)) .+ 1f-5)
    interaction_pae = dropdims(interaction_pae; dims=(2, 3))
    min_interaction_pae = minimum(pae .+ (1f0 .- target_designchain_inter_mask) .* 100000f0)

    is_design_token = design_mask_py
    target_design_inter_mask = token_pad_pair_mask .* (
        _unsqueeze_local(is_design_token, 3) .* _unsqueeze_local(is_target_token, 2) .+ _unsqueeze_local(is_target_token, 3) .* _unsqueeze_local(is_design_token, 2)
    )
    interaction_pae = sum(pae .* target_design_inter_mask; dims=(2, 3)) ./ (sum(target_design_inter_mask; dims=(2, 3)) .+ 1f-5)
    interaction_pae = dropdims(interaction_pae; dims=(2, 3))
    min_design_to_target_pae = minimum(pae .+ (1f0 .- target_design_inter_mask) .* 100000f0)

    resolved_logits_out = ch.token_level_confidence ? resolved_logits : resolved_logits_ff
    plddt_logits_out = ch.token_level_confidence ? plddt_logits : plddt_logits_ff

    out_dict = Dict{String,Any}(
        "pde_logits" => pde_logits,
        "plddt_logits" => plddt_logits_out,
        "resolved_logits" => resolved_logits_out,
        "pde" => permutedims(pde, (2, 3, 1)),
        "plddt" => plddt_ff,
        "complex_plddt" => complex_plddt,
        "complex_iplddt" => complex_iplddt,
        "complex_pde" => complex_pde,
        "complex_ipde" => complex_ipde,
        "interaction_pae" => interaction_pae,
        "min_design_to_target_pae" => sum(feats["design_mask"]) > 0 ? Float32[ min_design_to_target_pae ] : Float32[NaN],
        "min_interaction_pae" => Float32[ min_interaction_pae ],
        "protein_pae_value_weighted" => protein_pae_value_weighted,
        "ligand_pae_value_weighted" => ligand_pae_value_weighted,
        "interface_pae_value_weighted" => interface_pae_value_weighted,
        "protein_pae_value_unweighted" => protein_pae_value_unweighted,
        "ligand_pae_value_unweighted" => ligand_pae_value_unweighted,
        "interface_pae_value_unweighted" => interface_pae_value_unweighted,
        "pae_logits" => pae_logits,
        "pae" => permutedims(pae, (2, 3, 1)),
    )

    try
        ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm, design_to_target_iptm, design_iptm, design_iiptm, target_ptm, design_ptm =
            Onion.compute_ptms(pae_logits_py, permutedims(x_pred, (3, 2, 1)), feats, multiplicity)
        out_dict["ptm"] = ptm
        out_dict["iptm"] = iptm
        out_dict["ligand_iptm"] = ligand_iptm
        out_dict["protein_iptm"] = protein_iptm
        out_dict["pair_chains_iptm"] = pair_chains_iptm
        out_dict["design_to_target_iptm"] = design_to_target_iptm
        out_dict["design_iptm"] = design_iptm
        out_dict["design_iiptm"] = design_iiptm
        out_dict["target_ptm"] = target_ptm
        out_dict["design_ptm"] = design_ptm
    catch e
        out_dict["ptm"] = zeros(Float32, size(complex_plddt))
        out_dict["iptm"] = zeros(Float32, size(complex_plddt))
        out_dict["ligand_iptm"] = zeros(Float32, size(complex_plddt))
        out_dict["protein_iptm"] = zeros(Float32, size(complex_plddt))
        out_dict["pair_chains_iptm"] = Dict{Int,Dict{Int,Any}}()
        out_dict["design_to_target_iptm"] = zeros(Float32, size(complex_plddt))
        out_dict["design_iptm"] = zeros(Float32, size(complex_plddt))
        out_dict["design_iiptm"] = zeros(Float32, size(complex_plddt))
        out_dict["target_ptm"] = zeros(Float32, size(complex_plddt))
        out_dict["design_ptm"] = zeros(Float32, size(complex_plddt))
    end

    return out_dict
end
