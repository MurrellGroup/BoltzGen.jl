function _round_up_multiple(x::Int, k::Int)
    return ((x + k - 1) ÷ k) * k
end

function _onehot_vec(n::Int, idx::Int)
    v = zeros(Float32, n)
    if 1 <= idx <= n
        v[idx] = 1f0
    end
    return v
end

"""
Build minimal de novo atom14 features for protein-only sampling in feature-first layout.

This is a Julia-native fallback for pure inference workflows where Python featurization
is not available. It targets the same keys consumed by the current BoltzGen.jl
inference path and uses conservative defaults (single chain, all residues designed).
"""
function build_denovo_atom14_features(token_len::Int; batch::Int=1)
    @assert token_len > 0 "token_len must be positive"
    @assert batch > 0 "batch must be positive"

    T = token_len
    B = batch
    real_atoms = 14 * T
    M = _round_up_multiple(real_atoms, 32) # Atom encoder requires atom count divisible by window size.
    S = 1   # MSA rows
    U = 1   # Template rows

    # Token-level defaults
    token_pad_mask = ones(Float32, T, B)
    design_mask = ones(Int, T, B)
    chain_design_mask = ones(Float32, T, B)
    token_resolved_mask = ones(Float32, T, B)
    token_pair_mask = ones(Float32, T, B)
    token_disto_mask = ones(Float32, T, B)
    mol_type = fill(chain_type_ids["PROTEIN"], T, B)
    asym_id = zeros(Int, T, B)
    feature_asym_id = zeros(Int, T, B)
    entity_id = zeros(Int, T, B)
    sym_id = zeros(Int, T, B)
    cyclic = zeros(Int, T, B)
    token_index = reshape(repeat(collect(0:T-1), B), T, B)
    residue_index = copy(token_index)
    feature_residue_index = copy(token_index)

    # Token encodings
    gly_idx = token_ids["GLY"]
    gly_idx0 = token_ids0["GLY"]
    pad_idx = token_ids["<pad>"]
    res_type = zeros(Float32, num_tokens, T, B)
    profile = zeros(Float32, num_tokens, T, B)
    profile_affinity = zeros(Float32, num_tokens, T, B)
    for b in 1:B, t in 1:T
        res_type[gly_idx, t, b] = 1f0
        profile[gly_idx, t, b] = 1f0
        profile_affinity[gly_idx, t, b] = 1f0
    end

    deletion_mean = zeros(Float32, T, B)
    deletion_mean_affinity = zeros(Float32, T, B)
    msa = fill(gly_idx0, S, T, B)
    msa_mask = ones(Float32, S, T, B)
    msa_paired = zeros(Float32, S, T, B)
    has_deletion = zeros(Float32, S, T, B)
    deletion_value = zeros(Float32, S, T, B)
    target_msa_mask = zeros(Float32, T, B)

    method_feature = fill(1, T, B)
    modified = zeros(Int, T, B)
    binding_type = fill(binding_type_ids["UNSPECIFIED"], T, B)
    ss_type = fill(ss_type_ids["UNSPECIFIED"], T, B)
    ph_feature = fill(3, T, B)
    temp_feature = fill(3, T, B)

    # Pairwise/token features
    token_bonds = zeros(Float32, 1, T, T, B)
    type_bonds = zeros(Int, T, T, B)
    contact_threshold = zeros(Float32, T, T, B)
    contact_conditioning = zeros(Float32, length(contact_conditioning_info), T, T, B)
    contact_conditioning[contact_conditioning_info["UNSPECIFIED"] + 1, :, :, :] .= 1f0
    token_distance_mask = zeros(Float32, T, T, B)

    # Template features (masked-out)
    template_restype = zeros(Float32, num_tokens, T, U, B)
    template_restype[pad_idx, :, :, :] .= 1f0
    template_frame_rot = zeros(Float32, 3, 3, T, U, B)
    template_frame_t = zeros(Float32, 3, T, U, B)
    template_mask_frame = zeros(Float32, T, U, B)
    template_cb = zeros(Float32, 3, T, U, B)
    template_ca = zeros(Float32, 3, T, U, B)
    template_mask_cb = zeros(Float32, T, U, B)
    template_mask = zeros(Float32, T, U, B)

    # Atom-level features
    atom_pad_mask = zeros(Float32, M, B)
    atom_resolved_mask = zeros(Float32, M, B)
    backbone_mask = zeros(Float32, M, B)
    ref_pos = zeros(Float32, 3, M, B)
    coords = zeros(Float32, 3, M, B)
    center_coords = zeros(Float32, 3, T, B)
    ref_space_uid = zeros(Int, M, B)
    ref_charge = zeros(Float32, M, B)
    ref_element = zeros(Float32, num_elements, M, B)
    ref_atom_name_chars = zeros(Float32, 64, 4, M, B)
    atom_to_token = zeros(Float32, M, T, B)
    token_to_rep_atom = zeros(Float32, T, M, B)
    token_to_bb4_atoms = zeros(Float32, 4 * T, M, B)
    frames_idx = zeros(Int, T, 3, B)

    # Atomic number indices in the one-hot table
    c_idx = 6 + 1
    n_idx = 7 + 1
    o_idx = 8 + 1
    fake_lv_idx = 116 + 1

    for b in 1:B
        for t in 1:T
            base = 14 * (t - 1)
            active = base + 1:base + 14
            for (j, m) in enumerate(active)
                m <= M || continue
                atom_pad_mask[m, b] = 1f0
                atom_resolved_mask[m, b] = 1f0
                atom_to_token[m, t, b] = 1f0
                ref_space_uid[m, b] = t - 1
                if j == 1
                    ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars("N")
                    ref_element[n_idx, m, b] = 1f0
                    backbone_mask[m, b] = 1f0
                elseif j == 2
                    ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars("CA")
                    ref_element[c_idx, m, b] = 1f0
                    backbone_mask[m, b] = 1f0
                elseif j == 3
                    ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars("C")
                    ref_element[c_idx, m, b] = 1f0
                    backbone_mask[m, b] = 1f0
                elseif j == 4
                    ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars("O")
                    ref_element[o_idx, m, b] = 1f0
                    backbone_mask[m, b] = 1f0
                else
                    ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars("LV$(j-1)")
                    if 1 <= fake_lv_idx <= num_elements
                        ref_element[fake_lv_idx, m, b] = 1f0
                    end
                end
            end

            # Representative atom is CA (offset 2 in atom14)
            ca_m = base + 2
            if ca_m <= M
                token_to_rep_atom[t, ca_m, b] = 1f0
            end

            # Backbone atom gather matrix
            for k in 1:4
                m = base + k
                if m <= M
                    token_to_bb4_atoms[4 * (t - 1) + k, m, b] = 1f0
                end
            end

            # N, CA, C frame indices in 0-based atom indexing
            n0 = base
            frames_idx[t, 1, b] = n0
            frames_idx[t, 2, b] = n0 + 1
            frames_idx[t, 3, b] = n0 + 2
        end
    end

    masked_ref_atom_name_chars = build_masked_ref_atom_name_chars(atom_to_token, atom_pad_mask; mask_element="FL")

    return Dict(
        "atom_pad_mask" => atom_pad_mask,
        "ref_pos" => ref_pos,
        "ref_space_uid" => ref_space_uid,
        "ref_charge" => ref_charge,
        "ref_element" => ref_element,
        "ref_atom_name_chars" => ref_atom_name_chars,
        "masked_ref_atom_name_chars" => masked_ref_atom_name_chars,
        "atom_to_token" => atom_to_token,
        "res_type" => res_type,
        "profile" => profile,
        "deletion_mean" => deletion_mean,
        "profile_affinity" => profile_affinity,
        "deletion_mean_affinity" => deletion_mean_affinity,
        "token_pad_mask" => token_pad_mask,
        "mol_type" => mol_type,
        "affinity_token_mask" => zeros(Float32, T, B),
        "affinity_mw" => zeros(Float32, B),
        "token_bonds" => token_bonds,
        "type_bonds" => type_bonds,
        "contact_threshold" => contact_threshold,
        "contact_conditioning" => contact_conditioning,
        "feature_asym_id" => feature_asym_id,
        "feature_residue_index" => feature_residue_index,
        "entity_id" => entity_id,
        "token_index" => token_index,
        "sym_id" => sym_id,
        "cyclic" => cyclic,
        "asym_id" => asym_id,
        "template_restype" => template_restype,
        "template_frame_rot" => template_frame_rot,
        "template_frame_t" => template_frame_t,
        "template_mask_frame" => template_mask_frame,
        "template_cb" => template_cb,
        "template_ca" => template_ca,
        "template_mask_cb" => template_mask_cb,
        "template_mask" => template_mask,
        "token_distance_mask" => token_distance_mask,
        "center_coords" => center_coords,
        "coords" => coords,
        "token_to_bb4_atoms" => token_to_bb4_atoms,
        "token_to_rep_atom" => token_to_rep_atom,
        "frames_idx" => frames_idx,
        "atom_resolved_mask" => atom_resolved_mask,
        "backbone_mask" => backbone_mask,
        "chain_design_mask" => chain_design_mask,
        "design_mask" => design_mask,
        "msa" => msa,
        "has_deletion" => has_deletion,
        "deletion_value" => deletion_value,
        "msa_paired" => msa_paired,
        "msa_mask" => msa_mask,
        "target_msa_mask" => target_msa_mask,
        "method_feature" => method_feature,
        "modified" => modified,
        "binding_type" => binding_type,
        "ss_type" => ss_type,
        "ph_feature" => ph_feature,
        "temp_feature" => temp_feature,
        "residue_index" => residue_index,
        "token_pair_mask" => token_pair_mask,
        "token_resolved_mask" => token_resolved_mask,
        "token_disto_mask" => token_disto_mask,
    )
end
