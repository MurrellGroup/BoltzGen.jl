function boltz_masker(feats; mask::Bool=true, mask_backbone::Bool=false)
    mask || return feats

    new = copy(feats)

    token_pad_mask = feats["token_pad_mask"] .> 0.5
    design_mask = feats["design_mask"] .> 0.5
    token_mask = token_pad_mask .& design_mask # (T, B)

    atom_pad_mask = feats["atom_pad_mask"] .> 0.5
    atom_to_token = feats["atom_to_token"] # (M, T, B)
    design_mask_f = Float32.(design_mask)
    atom_design_mask = sum(atom_to_token .* reshape(design_mask_f, 1, size(design_mask_f,1), size(design_mask_f,2)); dims=2)
    atom_design_mask = dropdims(atom_design_mask; dims=2) .> 0.5 # (M, B)
    atom_mask = atom_pad_mask .& atom_design_mask
    if !mask_backbone && haskey(feats, "backbone_mask")
        atom_mask = atom_mask .& .!(feats["backbone_mask"] .> 0.5)
    end

    # Token feature masking
    if haskey(feats, "res_type")
        res_type = copy(feats["res_type"])
        unk_idx = token_ids["UNK"]
        mask_val = zeros(Float32, size(res_type, 1))
        mask_val[unk_idx] = 1f0
        mask_val = reshape(mask_val, size(res_type,1), 1, 1)
        token_mask_f = reshape(Float32.(token_mask), 1, size(token_mask,1), size(token_mask,2))
        res_type = res_type .* (1f0 .- token_mask_f) .+ mask_val .* token_mask_f
        new["res_type"] = res_type
    end

    if haskey(feats, "profile")
        profile = copy(feats["profile"])
        unk_idx = token_ids["UNK"]
        mask_val = zeros(Float32, size(profile, 1))
        mask_val[unk_idx] = 1f0
        mask_val = reshape(mask_val, size(profile,1), 1, 1)
        token_mask_f = reshape(Float32.(token_mask), 1, size(token_mask,1), size(token_mask,2))
        profile = profile .* (1f0 .- token_mask_f) .+ mask_val .* token_mask_f
        new["profile"] = profile
    end

    if haskey(feats, "deletion_mean")
        deletion_mean = copy(feats["deletion_mean"])
        deletion_mean = ifelse.(token_mask, 0f0, deletion_mean)
        new["deletion_mean"] = deletion_mean
    end

    if haskey(feats, "msa")
        msa = copy(feats["msa"])
        unk_id = token_ids0["UNK"]
        token_mask_b = reshape(token_mask, 1, size(token_mask,1), size(token_mask,2))
        msa = ifelse.(token_mask_b, unk_id, msa)
        if haskey(feats, "target_msa_mask")
            target_msa_mask = feats["target_msa_mask"] .> 0.5 # (T, B)
            if size(msa, 1) > 1
                target_mask_b = reshape(target_msa_mask, 1, size(target_msa_mask,1), size(target_msa_mask,2))
                gap_id = token_ids0["-"]
                msa_tail = view(msa, 2:size(msa,1), :, :)
                msa_tail .= ifelse.(target_mask_b, gap_id, msa_tail)
            end
        end
        new["msa"] = msa
    end

    if haskey(feats, "msa_mask")
        msa_mask = copy(feats["msa_mask"])
        if haskey(feats, "target_msa_mask")
            target_msa_mask = feats["target_msa_mask"] .> 0.5 # (T, B)
            if size(msa_mask, 1) > 1
                target_mask_b = reshape(target_msa_mask, 1, size(target_msa_mask,1), size(target_msa_mask,2))
                msa_tail = view(msa_mask, 2:size(msa_mask,1), :, :)
                msa_tail .= ifelse.(target_mask_b, 0f0, msa_tail)
            end
        else
            token_mask_b = reshape(token_mask, 1, size(token_mask,1), size(token_mask,2))
            msa_mask = ifelse.(token_mask_b, 0f0, msa_mask)
        end
        new["msa_mask"] = msa_mask
    end

    if haskey(feats, "msa_paired")
        msa_paired = copy(feats["msa_paired"])
        token_mask_b = reshape(token_mask, 1, size(token_mask,1), size(token_mask,2))
        msa_paired = ifelse.(token_mask_b, 0f0, msa_paired)
        new["msa_paired"] = msa_paired
    end

    if haskey(feats, "deletion_value")
        deletion_value = copy(feats["deletion_value"])
        token_mask_b = reshape(token_mask, 1, size(token_mask,1), size(token_mask,2))
        deletion_value = ifelse.(token_mask_b, 0f0, deletion_value)
        new["deletion_value"] = deletion_value
    end

    if haskey(feats, "has_deletion")
        has_deletion = copy(feats["has_deletion"])
        token_mask_b = reshape(token_mask, 1, size(token_mask,1), size(token_mask,2))
        has_deletion = ifelse.(token_mask_b, 0f0, has_deletion)
        new["has_deletion"] = has_deletion
    end

    # Atom feature masking
    if haskey(feats, "ref_element")
        ref_element = copy(feats["ref_element"])
        mask_val = zeros(Float32, size(ref_element, 1))
        mask_val[mask_element_index] = 1f0
        mask_val = reshape(mask_val, size(ref_element,1), 1, 1)
        atom_mask_f = reshape(Float32.(atom_mask), 1, size(atom_mask,1), size(atom_mask,2))
        ref_element = ref_element .* (1f0 .- atom_mask_f) .+ mask_val .* atom_mask_f
        new["ref_element"] = ref_element
    end

    if haskey(feats, "ref_charge")
        ref_charge = copy(feats["ref_charge"])
        ref_charge = ifelse.(atom_mask, 0f0, ref_charge)
        new["ref_charge"] = ref_charge
    end

    if haskey(feats, "ref_atom_name_chars") && haskey(feats, "masked_ref_atom_name_chars")
        ref_atom_name_chars = copy(feats["ref_atom_name_chars"])
        masked_ref_atom_name_chars = feats["masked_ref_atom_name_chars"]
        atom_mask_f = reshape(Float32.(atom_mask), 1, 1, size(atom_mask,1), size(atom_mask,2))
        ref_atom_name_chars = ref_atom_name_chars .* (1f0 .- atom_mask_f) .+ masked_ref_atom_name_chars .* atom_mask_f
        new["ref_atom_name_chars"] = ref_atom_name_chars
    end

    if haskey(feats, "ref_pos")
        ref_pos = copy(feats["ref_pos"])
        mask_ref_pos = atom_pad_mask .& atom_design_mask
        mask_ref_pos_f = reshape(Float32.(mask_ref_pos), 1, size(mask_ref_pos,1), size(mask_ref_pos,2))
        ref_pos = ref_pos .* (1f0 .- mask_ref_pos_f)
        new["ref_pos"] = ref_pos
    end

    if haskey(feats, "token_bonds")
        new["token_bonds"] = zeros(Float32, size(feats["token_bonds"]))
    end

    return new
end
