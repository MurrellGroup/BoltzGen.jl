function load_python_feature_npz(npz_path::AbstractString)
    npz = NPZ.npzread(npz_path)
    feats = Dict{String, Any}()

    get_arr(name) = npz[name]
    get_optional(name, default) = haskey(npz, name) ? npz[name] : default
    as_f32(x) = Float32.(x)
    as_i(x) = Int.(x)

    function squeeze_if_needed(arr)
        if ndims(arr) == 4 && size(arr, 2) == 1
            return dropdims(arr; dims=2)
        end
        return arr
    end

    feats["atom_pad_mask"] = as_f32(permutedims(get_arr("atom_pad_mask"), (2, 1)))
    feats["ref_pos"] = as_f32(permutedims(get_arr("ref_pos"), (3, 2, 1)))
    feats["ref_space_uid"] = as_i(permutedims(get_arr("ref_space_uid"), (2, 1)))
    feats["ref_charge"] = as_f32(permutedims(get_arr("ref_charge"), (2, 1)))
    feats["ref_element"] = as_f32(permutedims(get_arr("ref_element"), (3, 2, 1)))
    feats["ref_atom_name_chars"] = as_f32(permutedims(get_arr("ref_atom_name_chars"), (4, 3, 2, 1)))
    if haskey(npz, "masked_ref_atom_name_chars")
        feats["masked_ref_atom_name_chars"] = as_f32(permutedims(get_arr("masked_ref_atom_name_chars"), (4, 3, 2, 1)))
    end
    feats["atom_to_token"] = as_f32(permutedims(get_arr("atom_to_token"), (2, 3, 1)))

    feats["res_type"] = as_f32(permutedims(get_arr("res_type"), (3, 2, 1)))
    feats["profile"] = as_f32(permutedims(get_arr("profile"), (3, 2, 1)))
    feats["deletion_mean"] = as_f32(permutedims(get_arr("deletion_mean"), (2, 1)))
    profile_aff = get_optional("profile_affinity", get_arr("profile"))
    del_mean_aff = get_optional("deletion_mean_affinity", get_arr("deletion_mean"))
    feats["profile_affinity"] = as_f32(permutedims(profile_aff, (3, 2, 1)))
    feats["deletion_mean_affinity"] = as_f32(permutedims(del_mean_aff, (2, 1)))

    feats["token_pad_mask"] = as_f32(permutedims(get_arr("token_pad_mask"), (2, 1)))
    feats["mol_type"] = as_i(permutedims(get_arr("mol_type"), (2, 1)))
    if haskey(npz, "affinity_token_mask")
        feats["affinity_token_mask"] = as_f32(permutedims(get_arr("affinity_token_mask"), (2, 1)))
    else
        feats["affinity_token_mask"] = zeros(Float32, size(feats["token_pad_mask"]))
    end
    if haskey(npz, "affinity_mw")
        feats["affinity_mw"] = as_f32(get_arr("affinity_mw"))
    else
        feats["affinity_mw"] = zeros(Float32, size(feats["token_pad_mask"], 2))
    end

    feats["token_bonds"] = as_f32(permutedims(get_arr("token_bonds"), (4, 2, 3, 1)))
    feats["type_bonds"] = as_i(permutedims(get_arr("type_bonds"), (2, 3, 1)))
    feats["contact_threshold"] = as_f32(permutedims(get_arr("contact_threshold"), (2, 3, 1)))
    feats["contact_conditioning"] = as_f32(permutedims(get_arr("contact_conditioning"), (4, 2, 3, 1)))

    feats["feature_asym_id"] = as_i(permutedims(get_arr("feature_asym_id"), (2, 1)))
    feats["feature_residue_index"] = as_i(permutedims(get_arr("feature_residue_index"), (2, 1)))
    feats["residue_index"] = as_i(permutedims(get_arr("residue_index"), (2, 1)))
    feats["entity_id"] = as_i(permutedims(get_arr("entity_id"), (2, 1)))
    feats["token_index"] = as_i(permutedims(get_arr("token_index"), (2, 1)))
    feats["sym_id"] = as_i(permutedims(get_arr("sym_id"), (2, 1)))
    feats["cyclic"] = as_i(permutedims(get_arr("cyclic"), (2, 1)))
    feats["asym_id"] = as_i(permutedims(get_arr("asym_id"), (2, 1)))

    feats["template_restype"] = as_f32(permutedims(get_arr("template_restype"), (4, 3, 2, 1)))
    feats["template_frame_rot"] = as_f32(permutedims(get_arr("template_frame_rot"), (5, 4, 3, 2, 1)))
    feats["template_frame_t"] = as_f32(permutedims(get_arr("template_frame_t"), (4, 3, 2, 1)))
    feats["template_mask_frame"] = as_f32(permutedims(get_arr("template_mask_frame"), (3, 2, 1)))
    feats["template_cb"] = as_f32(permutedims(get_arr("template_cb"), (4, 3, 2, 1)))
    feats["template_ca"] = as_f32(permutedims(get_arr("template_ca"), (4, 3, 2, 1)))
    feats["template_mask_cb"] = as_f32(permutedims(get_arr("template_mask_cb"), (3, 2, 1)))
    feats["template_mask"] = as_f32(permutedims(get_arr("template_mask"), (3, 2, 1)))

    feats["token_distance_mask"] = as_f32(permutedims(get_arr("token_distance_mask"), (2, 3, 1)))
    feats["center_coords"] = as_f32(permutedims(get_arr("center_coords"), (3, 2, 1)))

    coords_raw = squeeze_if_needed(get_arr("coords"))
    feats["coords"] = as_f32(permutedims(coords_raw, (3, 2, 1)))

    token_to_bb4_raw = get_arr("token_to_bb4_atoms")
    token_to_bb4_raw = permutedims(token_to_bb4_raw, (3, 2, 4, 1))
    feats["token_to_bb4_atoms"] = as_f32(
        reshape(
            token_to_bb4_raw,
            size(token_to_bb4_raw, 1) * size(token_to_bb4_raw, 2),
            size(token_to_bb4_raw, 3),
            size(token_to_bb4_raw, 4),
        ),
    )
    feats["token_to_rep_atom"] = as_f32(permutedims(get_arr("token_to_rep_atom"), (2, 3, 1)))

    frames_idx_raw = squeeze_if_needed(get_arr("frames_idx"))
    feats["frames_idx"] = as_i(permutedims(frames_idx_raw, (2, 3, 1)))
    feats["atom_resolved_mask"] = as_f32(permutedims(get_arr("atom_resolved_mask"), (2, 1)))
    feats["backbone_mask"] = haskey(npz, "backbone_mask") ? as_f32(permutedims(get_arr("backbone_mask"), (2, 1))) : zeros(Float32, size(feats["atom_pad_mask"]))
    feats["chain_design_mask"] = as_f32(permutedims(get_arr("chain_design_mask"), (2, 1)))
    feats["design_mask"] = as_i(permutedims(get_arr("design_mask"), (2, 1)))

    feats["msa"] = as_i(permutedims(get_arr("msa"), (2, 3, 1)))
    feats["has_deletion"] = as_f32(permutedims(get_arr("has_deletion"), (2, 3, 1)))
    feats["deletion_value"] = as_f32(permutedims(get_arr("deletion_value"), (2, 3, 1)))
    feats["msa_paired"] = as_f32(permutedims(get_arr("msa_paired"), (2, 3, 1)))
    feats["msa_mask"] = as_f32(permutedims(get_arr("msa_mask"), (2, 3, 1)))
    feats["target_msa_mask"] = haskey(npz, "target_msa_mask") ? as_f32(permutedims(get_arr("target_msa_mask"), (2, 1))) : zeros(Float32, size(feats["token_pad_mask"]))

    feats["method_feature"] = as_i(permutedims(get_arr("method_feature"), (2, 1)))
    feats["modified"] = as_i(permutedims(get_arr("modified"), (2, 1)))
    feats["binding_type"] = as_i(permutedims(get_arr("binding_type"), (2, 1)))
    feats["ss_type"] = as_i(permutedims(get_arr("ss_type"), (2, 1)))
    feats["ph_feature"] = as_i(permutedims(get_arr("ph_feature"), (2, 1)))
    feats["temp_feature"] = as_i(permutedims(get_arr("temp_feature"), (2, 1)))

    # Present in newer featurizer outputs and used by pairformer masks.
    if haskey(npz, "token_pair_mask") && ndims(get_arr("token_pair_mask")) == 3
        feats["token_pair_mask"] = as_f32(permutedims(get_arr("token_pair_mask"), (2, 3, 1)))
    else
        t = size(feats["token_pad_mask"], 1)
        b = size(feats["token_pad_mask"], 2)
        pair = zeros(Float32, t, t, b)
        for ib in 1:b
            m = feats["token_pad_mask"][:, ib]
            pair[:, :, ib] .= m .* reshape(m, 1, :)
        end
        feats["token_pair_mask"] = pair
    end

    if haskey(npz, "token_resolved_mask")
        feats["token_resolved_mask"] = as_f32(permutedims(get_arr("token_resolved_mask"), (2, 1)))
    else
        feats["token_resolved_mask"] = copy(feats["token_pad_mask"])
    end

    if haskey(npz, "token_disto_mask")
        feats["token_disto_mask"] = as_f32(permutedims(get_arr("token_disto_mask"), (2, 1)))
    else
        feats["token_disto_mask"] = copy(feats["token_pad_mask"])
    end

    return feats
end
