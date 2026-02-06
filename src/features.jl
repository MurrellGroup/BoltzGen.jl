function _round_up_multiple(x::Int, k::Int)
    return ((x + k - 1) ÷ k) * k
end

function _onehot_set!(arr, idx::Int, t::Int, b::Int)
    if 1 <= idx <= size(arr, 1)
        arr[idx, t, b] = 1f0
    end
end

function _atomic_num_from_atom_name(atom_name::AbstractString)
    u = uppercase(strip(String(atom_name)))
    if startswith(u, "FL")
        return mask_element_id
    elseif startswith(u, "LV")
        return 116
    elseif startswith(u, "CL")
        return 17
    elseif startswith(u, "BR")
        return 35
    elseif startswith(u, "ZN")
        return 30
    elseif startswith(u, "MG")
        return 12
    elseif startswith(u, "FE")
        return 26
    elseif startswith(u, "NA")
        return 11
    elseif startswith(u, "CA") && !(u == "CA")
        # e.g. atom name "CA" in proteins means alpha carbon, not calcium.
        return 6
    end

    stripped = replace(u, r"[0-9]" => "")
    stripped = strip(stripped)
    if isempty(stripped)
        return 6
    elseif startswith(stripped, "C")
        return 6
    elseif startswith(stripped, "N")
        return 7
    elseif startswith(stripped, "O")
        return 8
    elseif startswith(stripped, "S")
        return 16
    elseif startswith(stripped, "P")
        return 15
    elseif startswith(stripped, "H")
        return 1
    end
    return 6
end

function tokens_from_sequence(sequence::AbstractString; chain_type::String="PROTEIN")
    seq = uppercase(strip(String(sequence)))
    chars = collect(seq)
    out = String[]

    if chain_type == "PROTEIN"
        for c in chars
            push!(out, get(prot_letter_to_token, c, "UNK"))
        end
    elseif chain_type == "DNA"
        for c in chars
            push!(out, get(dna_letter_to_token, c, "DN"))
        end
    elseif chain_type == "RNA"
        for c in chars
            push!(out, get(rna_letter_to_token, c, "N"))
        end
    else
        error("Unsupported chain_type: $chain_type")
    end

    return out
end

function _normalize_residue_token(token::AbstractString, mol_type_id::Int)
    t = uppercase(strip(String(token)))
    if haskey(token_ids, t)
        return t
    end
    if mol_type_id == chain_type_ids["PROTEIN"]
        if length(t) == 1
            return get(prot_letter_to_token, t[1], "UNK")
        end
        return "UNK"
    elseif mol_type_id == chain_type_ids["DNA"]
        if length(t) == 1
            return get(dna_letter_to_token, t[1], "DN")
        end
        return "DN"
    elseif mol_type_id == chain_type_ids["RNA"]
        if length(t) == 1
            return get(rna_letter_to_token, t[1], "N")
        end
        return "N"
    end
    return "UNK"
end

function _build_token_distance_mask(structure_group::Vector{Int})
    T = length(structure_group)
    out = zeros(Float32, T, T)
    if all(structure_group .== 1)
        out .= 1f0
        return out
    end
    for i in 1:T, j in 1:T
        if structure_group[i] > 0 && structure_group[i] == structure_group[j]
            out[i, j] = 1f0
        end
    end
    return out
end

function _propagate_chain_design_mask(design_mask::AbstractVector{Bool}, asym_id::Vector{Int}, bonds::Vector{NTuple{3,Int}})
    out = copy(design_mask)
    changed = true
    while changed
        changed = false
        chains = Set(asym_id[i] for i in eachindex(out) if out[i])
        for i in eachindex(out)
            if asym_id[i] in chains && !out[i]
                out[i] = true
                changed = true
            end
        end
        for (i, j, _) in bonds
            iok = 1 <= i <= length(out)
            jok = 1 <= j <= length(out)
            if iok && jok && (out[i] || out[j]) && !(out[i] && out[j])
                out[i] = true
                out[j] = true
                changed = true
            end
        end
    end
    return out
end

"""
Build conditioning-aware inference features in feature-first layout.

This path is Julia-native and does not depend on RDKit. It supports
protein/DNA/RNA token streams and the main conditioning channels needed
for inference (design masks, binding/ss labels, MSA mask, structure groups,
bonds, affinity token mask).
"""
function build_design_features(
    residue_tokens::Vector{String};
    mol_types::Union{Nothing,Vector{Int}}=nothing,
    asym_ids::Union{Nothing,Vector{Int}}=nothing,
    entity_ids::Union{Nothing,Vector{Int}}=nothing,
    sym_ids::Union{Nothing,Vector{Int}}=nothing,
    residue_indices::Union{Nothing,Vector{Int}}=nothing,
    design_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    chain_design_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    binding_type::Union{Nothing,Vector{Int}}=nothing,
    ss_type::Union{Nothing,Vector{Int}}=nothing,
    structure_group::Union{Nothing,Vector{Int}}=nothing,
    target_msa_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    affinity_token_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    affinity_mw::Float32=0f0,
    bonds::Vector{NTuple{3,Int}}=NTuple{3,Int}[],
    contact_pairs::Vector{NTuple{3,Any}}=NTuple{3,Any}[],
    batch::Int=1,
    pad_atom_multiple::Int=32,
    force_atom14_for_designed_protein::Bool=true,
)
    T = length(residue_tokens)
    T > 0 || error("residue_tokens cannot be empty")
    B = batch

    mol_types_v = mol_types === nothing ? fill(chain_type_ids["PROTEIN"], T) : copy(mol_types)
    length(mol_types_v) == T || error("mol_types length mismatch")

    tokens_norm = String[_normalize_residue_token(residue_tokens[i], mol_types_v[i]) for i in 1:T]

    asym_ids_v = asym_ids === nothing ? zeros(Int, T) : copy(asym_ids)
    entity_ids_v = entity_ids === nothing ? zeros(Int, T) : copy(entity_ids)
    sym_ids_v = sym_ids === nothing ? zeros(Int, T) : copy(sym_ids)
    residue_indices_v = residue_indices === nothing ? collect(0:T-1) : copy(residue_indices)
    design_mask_v = design_mask === nothing ? trues(T) : copy(design_mask)

    binding_v = binding_type === nothing ? fill(binding_type_ids["UNSPECIFIED"], T) : copy(binding_type)
    ss_v = ss_type === nothing ? fill(ss_type_ids["UNSPECIFIED"], T) : copy(ss_type)
    structure_group_v = structure_group === nothing ? fill(0, T) : copy(structure_group)
    target_msa_mask_v = target_msa_mask === nothing ? falses(T) : copy(target_msa_mask)
    affinity_token_mask_v = affinity_token_mask === nothing ? falses(T) : copy(affinity_token_mask)

    length(asym_ids_v) == T || error("asym_ids length mismatch")
    length(entity_ids_v) == T || error("entity_ids length mismatch")
    length(sym_ids_v) == T || error("sym_ids length mismatch")
    length(residue_indices_v) == T || error("residue_indices length mismatch")
    length(design_mask_v) == T || error("design_mask length mismatch")
    length(binding_v) == T || error("binding_type length mismatch")
    length(ss_v) == T || error("ss_type length mismatch")
    length(structure_group_v) == T || error("structure_group length mismatch")
    length(target_msa_mask_v) == T || error("target_msa_mask length mismatch")
    length(affinity_token_mask_v) == T || error("affinity_token_mask length mismatch")

    if chain_design_mask === nothing
        chain_design_mask_v = _propagate_chain_design_mask(design_mask_v, asym_ids_v, bonds)
    else
        chain_design_mask_v = copy(chain_design_mask)
    end
    length(chain_design_mask_v) == T || error("chain_design_mask length mismatch")

    token_atom_names = Vector{Vector{String}}(undef, T)
    token_atom_offsets = Vector{Int}(undef, T)
    atom_counter = 0
    for t in 1:T
        tok = tokens_norm[t]
        atoms = copy(get(ref_atoms, tok, ref_atoms["UNK"]))

        if force_atom14_for_designed_protein && design_mask_v[t] && mol_types_v[t] == chain_type_ids["PROTEIN"]
            while length(atoms) < 14
                push!(atoms, "LV$(length(atoms))")
            end
        end

        token_atom_offsets[t] = atom_counter + 1
        atom_counter += length(atoms)
        token_atom_names[t] = atoms
    end

    M_real = atom_counter
    M = _round_up_multiple(M_real, pad_atom_multiple)
    S = 1
    U = 1

    token_pad_mask = ones(Float32, T, B)
    design_mask_arr = zeros(Int, T, B)
    chain_design_mask_arr = zeros(Float32, T, B)
    token_resolved_mask = ones(Float32, T, B)
    token_pair_mask = ones(Float32, T, B)
    token_disto_mask = ones(Float32, T, B)
    mol_type = zeros(Int, T, B)
    asym_id = zeros(Int, T, B)
    feature_asym_id = zeros(Int, T, B)
    entity_id = zeros(Int, T, B)
    sym_id = zeros(Int, T, B)
    cyclic = zeros(Int, T, B)
    token_index = zeros(Int, T, B)
    residue_index = zeros(Int, T, B)
    feature_residue_index = zeros(Int, T, B)

    res_type = zeros(Float32, num_tokens, T, B)
    profile = zeros(Float32, num_tokens, T, B)
    profile_affinity = zeros(Float32, num_tokens, T, B)

    deletion_mean = zeros(Float32, T, B)
    deletion_mean_affinity = zeros(Float32, T, B)
    msa = zeros(Int, S, T, B)
    msa_mask = ones(Float32, S, T, B)
    msa_paired = zeros(Float32, S, T, B)
    has_deletion = zeros(Float32, S, T, B)
    deletion_value = zeros(Float32, S, T, B)
    target_msa_mask_arr = zeros(Float32, T, B)

    method_feature = fill(1, T, B)
    modified = zeros(Int, T, B)
    binding_type_arr = zeros(Int, T, B)
    ss_type_arr = zeros(Int, T, B)
    ph_feature = fill(3, T, B)
    temp_feature = fill(3, T, B)

    token_bonds = zeros(Float32, 1, T, T, B)
    type_bonds = zeros(Int, T, T, B)
    contact_threshold = zeros(Float32, T, T, B)
    contact_conditioning = zeros(Float32, length(contact_conditioning_info), T, T, B)
    contact_conditioning[contact_conditioning_info["UNSPECIFIED"] + 1, :, :, :] .= 1f0

    token_distance_mask_base = _build_token_distance_mask(structure_group_v)
    token_distance_mask = zeros(Float32, T, T, B)

    template_restype = zeros(Float32, num_tokens, T, U, B)
    template_restype[token_ids["<pad>"], :, :, :] .= 1f0
    template_frame_rot = zeros(Float32, 3, 3, T, U, B)
    template_frame_t = zeros(Float32, 3, T, U, B)
    template_mask_frame = zeros(Float32, T, U, B)
    template_cb = zeros(Float32, 3, T, U, B)
    template_ca = zeros(Float32, 3, T, U, B)
    template_mask_cb = zeros(Float32, T, U, B)
    template_mask = zeros(Float32, T, U, B)

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

    for b in 1:B
        for t in 1:T
            tok = tokens_norm[t]
            tok_idx = get(token_ids, tok, token_ids["UNK"])

            _onehot_set!(res_type, tok_idx, t, b)
            _onehot_set!(profile, tok_idx, t, b)
            _onehot_set!(profile_affinity, tok_idx, t, b)

            design_mask_arr[t, b] = design_mask_v[t] ? 1 : 0
            chain_design_mask_arr[t, b] = chain_design_mask_v[t] ? 1f0 : 0f0
            mol_type[t, b] = mol_types_v[t]
            asym_id[t, b] = asym_ids_v[t]
            feature_asym_id[t, b] = asym_ids_v[t]
            entity_id[t, b] = entity_ids_v[t]
            sym_id[t, b] = sym_ids_v[t]
            token_index[t, b] = t - 1
            residue_index[t, b] = residue_indices_v[t]
            feature_residue_index[t, b] = residue_indices_v[t]
            target_msa_mask_arr[t, b] = target_msa_mask_v[t] ? 1f0 : 0f0
            binding_type_arr[t, b] = binding_v[t]
            ss_type_arr[t, b] = ss_v[t]

            msa[1, t, b] = get(token_ids0, tok, token_ids0["UNK"])

            token_distance_mask[:, :, b] .= token_distance_mask_base

            atoms = token_atom_names[t]
            offset = token_atom_offsets[t]

            for (j, atom_name) in enumerate(atoms)
                m = offset + j - 1
                if m > M
                    continue
                end

                atom_pad_mask[m, b] = 1f0
                atom_resolved_mask[m, b] = 1f0
                atom_to_token[m, t, b] = 1f0
                ref_space_uid[m, b] = t - 1
                ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars(atom_name)

                z = _atomic_num_from_atom_name(atom_name)
                zidx = z + 1
                if 1 <= zidx <= num_elements
                    ref_element[zidx, m, b] = 1f0
                end

                is_bb = false
                if mol_types_v[t] == chain_type_ids["PROTEIN"]
                    is_bb = atom_name in protein_backbone_atom_names
                elseif mol_types_v[t] == chain_type_ids["DNA"] || mol_types_v[t] == chain_type_ids["RNA"]
                    is_bb = atom_name in nucleic_backbone_atom_names
                end
                if is_bb
                    backbone_mask[m, b] = 1f0
                end
            end

            rep_local = get(res_to_center_atom_id, tok, nothing)
            if rep_local === nothing || rep_local < 1 || rep_local > length(atoms)
                rep_local = min(2, length(atoms))
            end
            rep_m = offset + rep_local - 1
            if rep_m <= M
                token_to_rep_atom[t, rep_m, b] = 1f0
            end

            for k in 1:min(4, length(atoms))
                m = offset + k - 1
                if m <= M
                    token_to_bb4_atoms[4 * (t - 1) + k, m, b] = 1f0
                end
            end

            idx_n = findfirst(==("N"), atoms)
            idx_ca = findfirst(==("CA"), atoms)
            idx_c = findfirst(==("C"), atoms)

            if idx_n === nothing
                idx_n = 1
            end
            if idx_ca === nothing
                idx_ca = min(2, length(atoms))
            end
            if idx_c === nothing
                idx_c = min(3, length(atoms))
            end

            frames_idx[t, 1, b] = offset + idx_n - 2
            frames_idx[t, 2, b] = offset + idx_ca - 2
            frames_idx[t, 3, b] = offset + idx_c - 2
        end

        for (i, j, bt) in bonds
            if 1 <= i <= T && 1 <= j <= T
                token_bonds[1, i, j, b] = 1f0
                token_bonds[1, j, i, b] = 1f0
                type_bonds[i, j, b] = bt
                type_bonds[j, i, b] = bt
            end
        end

        for (i, j, ctype_any) in contact_pairs
            if !(1 <= i <= T && 1 <= j <= T)
                continue
            end
            ctype = string(ctype_any)
            if haskey(contact_conditioning_info, ctype)
                contact_conditioning[:, i, j, b] .= 0f0
                contact_conditioning[contact_conditioning_info[ctype] + 1, i, j, b] = 1f0
            end
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
        "affinity_token_mask" => reshape(Float32.(affinity_token_mask_v), T, 1) .* ones(Float32, 1, B),
        "affinity_mw" => fill(Float32(affinity_mw), B),
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
        "chain_design_mask" => chain_design_mask_arr,
        "design_mask" => design_mask_arr,
        "msa" => msa,
        "has_deletion" => has_deletion,
        "deletion_value" => deletion_value,
        "msa_paired" => msa_paired,
        "msa_mask" => msa_mask,
        "target_msa_mask" => target_msa_mask_arr,
        "method_feature" => method_feature,
        "modified" => modified,
        "binding_type" => binding_type_arr,
        "ss_type" => ss_type_arr,
        "ph_feature" => ph_feature,
        "temp_feature" => temp_feature,
        "residue_index" => residue_index,
        "token_pair_mask" => token_pair_mask,
        "token_resolved_mask" => token_resolved_mask,
        "token_disto_mask" => token_disto_mask,
        "structure_group" => reshape(structure_group_v, T, 1) .* ones(Int, 1, B),
    )
end

function build_denovo_atom14_features(token_len::Int; batch::Int=1)
    token_len > 0 || error("token_len must be positive")
    residues = fill("GLY", token_len)
    return build_design_features(residues; batch=batch)
end

function build_denovo_atom14_features_from_sequence(sequence::AbstractString; chain_type::String="PROTEIN", batch::Int=1)
    residues = tokens_from_sequence(sequence; chain_type=chain_type)
    return build_design_features(residues; batch=batch)
end
