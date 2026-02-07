function _round_up_multiple(x::Int, k::Int)
    return ((x + k - 1) ÷ k) * k
end

function _onehot_set!(arr, idx::Int, t::Int, b::Int)
    if 1 <= idx <= size(arr, 1)
        arr[idx, t, b] = 1f0
    end
end

function _atomic_num_from_atom_name(atom_name::AbstractString)
    raw = strip(String(atom_name))
    u = uppercase(raw)
    if startswith(u, "FL")
        return mask_element_id
    elseif startswith(u, "LV")
        return 116
    end

    # Prefer chemistry-style symbols when present in atom names, especially
    # for SMILES-derived ligands (e.g. "Cl1", "Br2", "Se3").
    letters = filter(isletter, raw)
    if !isempty(letters)
        if length(letters) >= 2 && islowercase(letters[2])
            sym2 = string(uppercase(letters[1]), lowercase(letters[2]))
            if sym2 == "Cl"
                return 17
            elseif sym2 == "Br"
                return 35
            elseif sym2 == "Si"
                return 14
            elseif sym2 == "Se"
                return 34
            elseif sym2 == "Na"
                return 11
            elseif sym2 == "Mg"
                return 12
            elseif sym2 == "Zn"
                return 30
            elseif sym2 == "Fe"
                return 26
            elseif sym2 == "Ca"
                return 20
            elseif sym2 == "Cu"
                return 29
            elseif sym2 == "Mn"
                return 25
            elseif sym2 == "Co"
                return 27
            elseif sym2 == "Ni"
                return 28
            elseif sym2 == "Li"
                return 3
            elseif sym2 == "Al"
                return 13
            elseif sym2 == "Sn"
                return 50
            elseif sym2 == "Ag"
                return 47
            elseif sym2 == "Pt"
                return 78
            elseif sym2 == "Hg"
                return 80
            elseif sym2 == "Pb"
                return 82
            end
        end
    end

    # Uppercase fallback for sources that drop mixed-case element symbols.
    if startswith(u, "CL")
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
    elseif startswith(u, "CU")
        return 29
    elseif startswith(u, "MN")
        return 25
    elseif startswith(u, "CO")
        return 27
    elseif startswith(u, "NI")
        return 28
    elseif startswith(u, "CA") && !(u == "CA")
        # e.g. atom name "CA" in proteins means alpha carbon, not calcium.
        return 6
    end

    stripped = replace(u, r"[^A-Z]" => "")
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
    elseif startswith(stripped, "F")
        return 9
    elseif startswith(stripped, "I")
        return 53
    elseif startswith(stripped, "B")
        return 5
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

function _msa_char_to_token(c::Char, mol_type_id::Int)
    if c == '-'
        return "-"
    end

    uc = uppercase(c)
    if mol_type_id == chain_type_ids["PROTEIN"]
        return get(prot_letter_to_token, uc, "UNK")
    elseif mol_type_id == chain_type_ids["DNA"]
        return get(dna_letter_to_token, uc, "DN")
    elseif mol_type_id == chain_type_ids["RNA"]
        return get(rna_letter_to_token, uc, "N")
    end
    return "UNK"
end

function _normalize_msa_row(row::AbstractString, T::Int)
    aligned = Char[]
    has_del = zeros(Float32, T)
    del_val = zeros(Float32, T)

    for c0 in String(row)
        if isspace(c0)
            continue
        elseif c0 == '.'
            # HHblits/A3M commonly uses '.' as a gap-like placeholder.
            push!(aligned, '-')
            continue
        elseif islowercase(c0)
            # Lowercase in A3M represents insertions relative to query.
            if !isempty(aligned)
                pos = min(length(aligned), T)
                has_del[pos] = 1f0
                del_val[pos] += 1f0
            end
            continue
        end

        c = uppercase(c0)
        if c == '*'
            c = 'X'
        end
        push!(aligned, c)
    end

    length(aligned) == T || error("MSA row has aligned length $(length(aligned)) but expected $T")
    return String(aligned), has_del, del_val
end

function load_msa_sequences(path::AbstractString; max_rows::Union{Nothing, Int}=nothing)
    isfile(path) || error("MSA file not found: $path")
    lines = readlines(path)

    rows = String[]
    has_fasta_header = any(startswith(strip(line), ">") for line in lines)

    if has_fasta_header
        current = IOBuffer()
        in_record = false
        for raw in lines
            line = strip(raw)
            isempty(line) && continue
            if startswith(line, ">")
                if in_record
                    push!(rows, String(take!(current)))
                end
                in_record = true
                continue
            end
            in_record = true
            print(current, line)
        end
        if in_record && position(current) > 0
            push!(rows, String(take!(current)))
        end
    else
        for raw in lines
            line = strip(raw)
            isempty(line) && continue
            push!(rows, line)
        end
    end

    isempty(rows) && error("No MSA rows found in file: $path")
    if max_rows !== nothing
        max_rows > 0 || error("max_rows must be positive")
        rows = rows[1:min(end, max_rows)]
    end
    return rows
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

function _find_atom_idx_or_default(atoms::Vector{String}, atom_name::String, default_idx::Int=1)
    idx = findfirst(==(atom_name), atoms)
    if idx === nothing
        return clamp(default_idx, 1, max(length(atoms), 1))
    end
    return Int(idx)
end

function _frame_atom_indices(tok::String, mol_type_id::Int, atoms::Vector{String})
    n = max(length(atoms), 1)
    idx_a = 1
    idx_b = 1
    idx_c = 1
    idx_d = 1

    if length(atoms) < 3 || tok in ("<pad>", "UNK", "-", "DN", "N")
        return idx_a, idx_b, idx_c, idx_d
    end

    if mol_type_id == chain_type_ids["PROTEIN"]
        if haskey(ref_atoms, tok)
            idx_a = _find_atom_idx_or_default(atoms, "N", 1)
            idx_b = _find_atom_idx_or_default(atoms, "CA", min(2, n))
            idx_c = _find_atom_idx_or_default(atoms, "C", min(3, n))
            idx_d = _find_atom_idx_or_default(atoms, "O", idx_c)
        else
            idx_a = _find_atom_idx_or_default(atoms, "CA", 1)
            idx_b = _find_atom_idx_or_default(atoms, "N", idx_a)
            idx_c = _find_atom_idx_or_default(atoms, "C", idx_b)
            idx_d = idx_a
        end
        return idx_a, idx_b, idx_c, idx_d
    end

    if mol_type_id == chain_type_ids["DNA"] || mol_type_id == chain_type_ids["RNA"]
        idx_a = _find_atom_idx_or_default(atoms, "C1'", 1)
        idx_b = _find_atom_idx_or_default(atoms, "C3'", idx_a)
        idx_c = _find_atom_idx_or_default(atoms, "C4'", idx_b)
        idx_d = _find_atom_idx_or_default(atoms, "P", idx_a)
        return idx_a, idx_b, idx_c, idx_d
    end

    return idx_a, idx_b, idx_c, idx_d
end

function _safe_xyz(atom_coords::Dict{String, NTuple{3, Float32}}, atom_name::String)
    if haskey(atom_coords, atom_name)
        return atom_coords[atom_name], true
    end
    return (0f0, 0f0, 0f0), false
end

function _norm3(v::NTuple{3, Float32})
    return sqrt(v[1]^2 + v[2]^2 + v[3]^2)
end

function _normalize3(v::NTuple{3, Float32})
    n = _norm3(v)
    if n <= 1f-8
        return (0f0, 0f0, 0f0), false
    end
    return (v[1] / n, v[2] / n, v[3] / n), true
end

function _cross3(a::NTuple{3, Float32}, b::NTuple{3, Float32})
    return (
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
end

function _sub3(a::NTuple{3, Float32}, b::NTuple{3, Float32})
    return (a[1] - b[1], a[2] - b[2], a[3] - b[3])
end

function _template_frame_from_points(a::NTuple{3, Float32}, b::NTuple{3, Float32}, c::NTuple{3, Float32})
    # Build a right-handed local frame with origin at b.
    # Columns are basis vectors in global coordinates.
    e1, ok1 = _normalize3(_sub3(c, b))
    v2 = _sub3(a, b)
    e3_raw = _cross3(e1, v2)
    e3, ok3 = _normalize3(e3_raw)
    e2 = _cross3(e3, e1)
    if !(ok1 && ok3)
        return Matrix{Float32}(I, 3, 3), b, false
    end
    rot = zeros(Float32, 3, 3)
    rot[1, 1] = e1[1]; rot[2, 1] = e1[2]; rot[3, 1] = e1[3]
    rot[1, 2] = e2[1]; rot[2, 2] = e2[2]; rot[3, 2] = e2[3]
    rot[1, 3] = e3[1]; rot[2, 3] = e3[2]; rot[3, 3] = e3[3]
    return rot, b, true
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
    cyclic_period::Union{Nothing,Vector{Int}}=nothing,
    residue_indices::Union{Nothing,Vector{Int}}=nothing,
    design_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    chain_design_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    binding_type::Union{Nothing,Vector{Int}}=nothing,
    ss_type::Union{Nothing,Vector{Int}}=nothing,
    structure_group::Union{Nothing,Vector{Int}}=nothing,
    target_msa_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    msa_sequences::Union{Nothing,Vector{String}}=nothing,
    msa_paired_rows::Union{Nothing,AbstractVector{Bool}}=nothing,
    msa_mask_rows::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    msa_has_deletion_rows::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    msa_deletion_value_rows::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    max_msa_rows::Union{Nothing,Int}=nothing,
    affinity_token_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    affinity_mw::Float32=0f0,
    template_paths::Vector{String}=String[],
    max_templates::Union{Nothing,Int}=nothing,
    template_include_chains::Union{Nothing,Vector{String}}=nothing,
    bonds::Vector{NTuple{3,Int}}=NTuple{3,Int}[],
    contact_pairs::Vector{NTuple{3,Any}}=NTuple{3,Any}[],
    batch::Int=1,
    pad_atom_multiple::Int=32,
    force_atom14_for_designed_protein::Bool=true,
    token_atom_names_override::Union{Nothing,Vector{Vector{String}}}=nothing,
    token_atom_coords_override::Union{Nothing,Vector{Dict{String,NTuple{3,Float32}}}}=nothing,
    center_coords_override::Union{Nothing,Vector{NTuple{3,Float32}}}=nothing,
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
    cyclic_period_v = cyclic_period === nothing ? zeros(Int, T) : copy(cyclic_period)
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
    length(cyclic_period_v) == T || error("cyclic_period length mismatch")
    length(residue_indices_v) == T || error("residue_indices length mismatch")
    length(design_mask_v) == T || error("design_mask length mismatch")
    length(binding_v) == T || error("binding_type length mismatch")
    length(ss_v) == T || error("ss_type length mismatch")
    length(structure_group_v) == T || error("structure_group length mismatch")
    length(target_msa_mask_v) == T || error("target_msa_mask length mismatch")
    length(affinity_token_mask_v) == T || error("affinity_token_mask length mismatch")
    token_atom_names_override === nothing || length(token_atom_names_override) == T || error("token_atom_names_override length mismatch")
    token_atom_coords_override === nothing || length(token_atom_coords_override) == T || error("token_atom_coords_override length mismatch")
    center_coords_override === nothing || length(center_coords_override) == T || error("center_coords_override length mismatch")

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
        is_nonpoly = mol_types_v[t] == chain_type_ids["NONPOLYMER"]
        if is_nonpoly
            token_atom_names_override === nothing && error("NONPOLYMER token at index $t requires token_atom_names_override")
            token_atom_coords_override === nothing && error("NONPOLYMER token at index $t requires token_atom_coords_override")
            isempty(token_atom_names_override[t]) && error("NONPOLYMER token at index $t has empty atom-name override")
        end
        use_override_atoms = token_atom_names_override !== nothing &&
            !isempty(token_atom_names_override[t]) &&
            !(force_atom14_for_designed_protein && design_mask_v[t] && mol_types_v[t] == chain_type_ids["PROTEIN"])
        atoms = if use_override_atoms
            copy(token_atom_names_override[t])
        else
            copy(get(ref_atoms, tok, ref_atoms["UNK"]))
        end

        if force_atom14_for_designed_protein && design_mask_v[t] && mol_types_v[t] == chain_type_ids["PROTEIN"]
            while length(atoms) < 14
                push!(atoms, "LV$(length(atoms))")
            end
        end

        token_atom_offsets[t] = atom_counter + 1
        atom_counter += length(atoms)
        token_atom_names[t] = atoms
    end

    # Match python featurizer semantics: atoms belonging to the same
    # (chain/asym, residue_index) share a residue-space uid.
    token_space_uid = zeros(Int, T)
    chain_res_to_space_uid = Dict{Tuple{Int,Int},Int}()
    next_space_uid = 0
    for t in 1:T
        key = (asym_ids_v[t], residue_indices_v[t])
        if !haskey(chain_res_to_space_uid, key)
            chain_res_to_space_uid[key] = next_space_uid
            next_space_uid += 1
        end
        token_space_uid[t] = chain_res_to_space_uid[key]
    end

    M_real = atom_counter
    M = _round_up_multiple(M_real, pad_atom_multiple)

    msa_rows = msa_sequences === nothing ? nothing : copy(msa_sequences)
    if msa_rows !== nothing
        if max_msa_rows !== nothing
            max_msa_rows > 0 || error("max_msa_rows must be positive")
            msa_rows = msa_rows[1:min(end, max_msa_rows)]
        end
        isempty(msa_rows) && error("msa_sequences is empty after trimming")
    end
    S = msa_rows === nothing ? 1 : length(msa_rows)

    template_paths_sel = copy(template_paths)
    if max_templates !== nothing
        max_templates > 0 || error("max_templates must be positive")
        template_paths_sel = template_paths_sel[1:min(end, max_templates)]
    end
    template_parsed = Any[]
    for p in template_paths_sel
        push!(template_parsed, load_structure_tokens(p; include_chains=template_include_chains, include_nonpolymer=false))
    end
    U = isempty(template_parsed) ? 1 : length(template_parsed)

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
    msa_paired = ones(Float32, S, T, B)
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

    msa_ids_input = zeros(Int, S, T)
    msa_mask_input = ones(Float32, S, T)
    msa_paired_input = ones(Float32, S, T)
    msa_has_deletion_input = zeros(Float32, S, T)
    msa_deletion_value_input = zeros(Float32, S, T)

    if msa_rows === nothing
        for t in 1:T
            msa_ids_input[1, t] = get(token_ids0, tokens_norm[t], token_ids0["UNK"])
        end
    else
        for s in 1:S
            row_norm, has_del_row, del_val_row = _normalize_msa_row(msa_rows[s], T)
            for t in 1:T
                tok = _msa_char_to_token(row_norm[t], mol_types_v[t])
                msa_ids_input[s, t] = get(token_ids0, tok, token_ids0["UNK"])
                msa_has_deletion_input[s, t] = has_del_row[t]
                msa_deletion_value_input[s, t] = del_val_row[t]
            end
        end
    end

    if msa_paired_rows !== nothing
        length(msa_paired_rows) == S || error("msa_paired_rows length mismatch")
        for s in 1:S
            v = msa_paired_rows[s] ? 1f0 : 0f0
            msa_paired_input[s, :] .= v
        end
    end
    if msa_mask_rows !== nothing
        size(msa_mask_rows, 1) == S || error("msa_mask_rows row mismatch")
        size(msa_mask_rows, 2) == T || error("msa_mask_rows token count mismatch")
        msa_mask_input .= Float32.(msa_mask_rows)
    end
    if msa_has_deletion_rows !== nothing
        size(msa_has_deletion_rows, 1) == S || error("msa_has_deletion_rows row mismatch")
        size(msa_has_deletion_rows, 2) == T || error("msa_has_deletion_rows token count mismatch")
        msa_has_deletion_input .= Float32.(msa_has_deletion_rows)
    end
    if msa_deletion_value_rows !== nothing
        size(msa_deletion_value_rows, 1) == S || error("msa_deletion_value_rows row mismatch")
        size(msa_deletion_value_rows, 2) == T || error("msa_deletion_value_rows token count mismatch")
        msa_deletion_value_input .= Float32.(msa_deletion_value_rows)
    end

    for b in 1:B
        msa[:, :, b] .= msa_ids_input
        msa_mask[:, :, b] .= msa_mask_input
        msa_paired[:, :, b] .= msa_paired_input
        has_deletion[:, :, b] .= msa_has_deletion_input
        deletion_value[:, :, b] .= msa_deletion_value_input

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
            cyclic[t, b] = cyclic_period_v[t]
            token_index[t, b] = t - 1
            residue_index[t, b] = residue_indices_v[t]
            feature_residue_index[t, b] = residue_indices_v[t]
            target_msa_mask_arr[t, b] = target_msa_mask_v[t] ? 1f0 : 0f0
            binding_type_arr[t, b] = binding_v[t]
            ss_type_arr[t, b] = ss_v[t]

            token_distance_mask[:, :, b] .= token_distance_mask_base

            atoms = token_atom_names[t]
            offset = token_atom_offsets[t]
            atom_coord_override = token_atom_coords_override === nothing ? nothing : token_atom_coords_override[t]
            is_nonpoly = mol_types_v[t] == chain_type_ids["NONPOLYMER"]

            for (j, atom_name) in enumerate(atoms)
                m = offset + j - 1
                if m > M
                    continue
                end

                atom_pad_mask[m, b] = 1f0
                atom_resolved_mask[m, b] = 1f0
                atom_to_token[m, t, b] = 1f0
                ref_space_uid[m, b] = token_space_uid[t]
                ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars(atom_name)
                if atom_coord_override !== nothing && haskey(atom_coord_override, atom_name)
                    xyz = atom_coord_override[atom_name]
                    ref_pos[1, m, b] = xyz[1]
                    ref_pos[2, m, b] = xyz[2]
                    ref_pos[3, m, b] = xyz[3]
                    coords[1, m, b] = xyz[1]
                    coords[2, m, b] = xyz[2]
                    coords[3, m, b] = xyz[3]
                elseif is_nonpoly
                    error("Missing coordinate override for NONPOLYMER atom '$atom_name' at token index $t")
                elseif haskey(ref_atom_pos, tok)
                    tok_ref_pos = ref_atom_pos[tok]
                    if haskey(tok_ref_pos, atom_name)
                        xyz = tok_ref_pos[atom_name]
                        ref_pos[1, m, b] = xyz[1]
                        ref_pos[2, m, b] = xyz[2]
                        ref_pos[3, m, b] = xyz[3]
                    end
                end
                if haskey(ref_atom_charge, tok)
                    tok_ref_charge = ref_atom_charge[tok]
                    if haskey(tok_ref_charge, atom_name)
                        ref_charge[m, b] = tok_ref_charge[atom_name]
                    end
                end

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

            idx_a, idx_b, idx_c, idx_d = _frame_atom_indices(tok, mol_types_v[t], atoms)
            bb4 = (idx_a, idx_b, idx_c, idx_d)
            for k in 1:4
                m = offset + bb4[k] - 1
                if m <= M
                    token_to_bb4_atoms[4 * (t - 1) + k, m, b] = 1f0
                end
            end

            if center_coords_override !== nothing
                ctr = center_coords_override[t]
                center_coords[1, t, b] = ctr[1]
                center_coords[2, t, b] = ctr[2]
                center_coords[3, t, b] = ctr[3]
            elseif atom_coord_override !== nothing && rep_m <= M
                center_coords[1, t, b] = ref_pos[1, rep_m, b]
                center_coords[2, t, b] = ref_pos[2, rep_m, b]
                center_coords[3, t, b] = ref_pos[3, rep_m, b]
            end

            frames_idx[t, 1, b] = offset + idx_a - 2
            frames_idx[t, 2, b] = offset + idx_b - 2
            frames_idx[t, 3, b] = offset + idx_c - 2
        end

        # Match python's compute_frames_nonpolymer: for nonpolymer chains with
        # >=3 atoms, compute per-token frames from nearest-neighbor triplets
        # within the chain atom set.
        chain_order = unique(asym_ids_v)
        token_cursor = 1
        for chain_id in chain_order
            chain_tokens = findall(x -> x == chain_id, asym_ids_v)
            isempty(chain_tokens) && continue
            first(chain_tokens) == token_cursor || error("asym_id tokens must be contiguous for chain $chain_id")
            last(chain_tokens) == token_cursor + length(chain_tokens) - 1 || error("asym_id tokens must be contiguous for chain $chain_id")

            t_start = token_cursor
            t_end = token_cursor + length(chain_tokens) - 1
            token_cursor = t_end + 1

            num_atoms_chain = sum(length(token_atom_names[t]) for t in t_start:t_end)
            if mol_types_v[t_start] != chain_type_ids["NONPOLYMER"] || num_atoms_chain < 3
                continue
            end
            num_atoms_chain == length(chain_tokens) || error(
                "NONPOLYMER frame recompute requires one atom per token for chain $chain_id; got tokens=$(length(chain_tokens)) atoms=$num_atoms_chain",
            )

            atom_ids = Vector{Int}(undef, num_atoms_chain)
            for t in t_start:t_end
                atom_ids[t - t_start + 1] = token_atom_offsets[t]
            end

            dist = fill(Inf32, num_atoms_chain, num_atoms_chain)
            for i in 1:num_atoms_chain
                mi = atom_ids[i]
                ri = atom_resolved_mask[mi, b] > 0.5f0
                xi = ref_pos[1, mi, b]
                yi = ref_pos[2, mi, b]
                zi = ref_pos[3, mi, b]
                for j in 1:num_atoms_chain
                    mj = atom_ids[j]
                    rj = atom_resolved_mask[mj, b] > 0.5f0
                    if ri && rj
                        dx = xi - ref_pos[1, mj, b]
                        dy = yi - ref_pos[2, mj, b]
                        dz = zi - ref_pos[3, mj, b]
                        dist[i, j] = sqrt(dx * dx + dy * dy + dz * dz)
                    end
                end
            end

            for li in 1:num_atoms_chain
                # Keep neighbor order deterministic under equal/near-equal distances.
                # Python argsort behavior effectively prefers lower indices on ties.
                ord = collect(1:num_atoms_chain)
                sort!(ord; by=j -> (dist[li, j], j))
                length(ord) >= 3 || error("Failed to compute nonpolymer frame neighbors for chain $chain_id")
                t = t_start + li - 1
                frames_idx[t, 1, b] = atom_ids[ord[2]] - 1
                frames_idx[t, 2, b] = atom_ids[ord[1]] - 1
                frames_idx[t, 3, b] = atom_ids[ord[3]] - 1
            end
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

        if !isempty(template_parsed)
            for (u, tmpl_any) in enumerate(template_parsed)
                tmpl = tmpl_any
                L = min(T, length(tmpl.residue_tokens))
                for t in 1:L
                    tok_u = _normalize_residue_token(tmpl.residue_tokens[t], tmpl.mol_types[t])
                    tok_idx_u = get(token_ids, tok_u, token_ids["UNK"])
                    template_restype[:, t, u, b] .= 0f0
                    _onehot_set!(template_restype[:, :, u, :], tok_idx_u, t, b)
                    template_mask[t, u, b] = 1f0

                    atoms_u = tmpl.token_atom_names[t]
                    coords_u = tmpl.token_atom_coords[t]
                    mol_u = tmpl.mol_types[t]

                    ca_name = if mol_u == chain_type_ids["PROTEIN"]
                        "CA"
                    elseif mol_u == chain_type_ids["DNA"] || mol_u == chain_type_ids["RNA"]
                        "C1'"
                    else
                        isempty(atoms_u) ? "" : atoms_u[1]
                    end
                    cb_name = get(res_to_disto_atom, tok_u, ca_name)

                    ca_xyz, has_ca = _safe_xyz(coords_u, ca_name)
                    cb_xyz, has_cb = _safe_xyz(coords_u, cb_name)
                    if has_ca
                        template_ca[1, t, u, b] = ca_xyz[1]
                        template_ca[2, t, u, b] = ca_xyz[2]
                        template_ca[3, t, u, b] = ca_xyz[3]
                    end
                    if has_cb
                        template_cb[1, t, u, b] = cb_xyz[1]
                        template_cb[2, t, u, b] = cb_xyz[2]
                        template_cb[3, t, u, b] = cb_xyz[3]
                        template_mask_cb[t, u, b] = 1f0
                    elseif has_ca
                        template_cb[1, t, u, b] = ca_xyz[1]
                        template_cb[2, t, u, b] = ca_xyz[2]
                        template_cb[3, t, u, b] = ca_xyz[3]
                        template_mask_cb[t, u, b] = 1f0
                    end

                    if length(atoms_u) >= 3
                        idx_a, idx_b, idx_c, _ = _frame_atom_indices(tok_u, mol_u, atoms_u)
                        name_a = atoms_u[idx_a]
                        name_b = atoms_u[idx_b]
                        name_c = atoms_u[idx_c]
                        xyz_a, has_a = _safe_xyz(coords_u, name_a)
                        xyz_b, has_b = _safe_xyz(coords_u, name_b)
                        xyz_c, has_c = _safe_xyz(coords_u, name_c)
                        if has_a && has_b && has_c
                            rot, trans, ok_frame = _template_frame_from_points(xyz_a, xyz_b, xyz_c)
                            if ok_frame
                                template_frame_rot[:, :, t, u, b] .= rot
                                template_frame_t[1, t, u, b] = trans[1]
                                template_frame_t[2, t, u, b] = trans[2]
                                template_frame_t[3, t, u, b] = trans[3]
                                template_mask_frame[t, u, b] = 1f0
                            end
                        end
                    end
                end
            end
        end

        for t in 1:T
            profile[:, t, b] .= 0f0
            denom = 0f0
            del_sum = 0f0
            for s in 1:S
                w = msa_mask[s, t, b]
                if w <= 0f0
                    continue
                end
                idx = msa[s, t, b] + 1
                if 1 <= idx <= num_tokens
                    profile[idx, t, b] += w
                end
                del_sum += deletion_value[s, t, b] * w
                denom += w
            end
            if denom > 1f-8
                profile[:, t, b] ./= denom
                deletion_mean[t, b] = del_sum / denom
            else
                tok_idx = get(token_ids, tokens_norm[t], token_ids["UNK"])
                _onehot_set!(profile, tok_idx, t, b)
                deletion_mean[t, b] = 0f0
            end
            profile_affinity[:, t, b] .= profile[:, t, b]
            deletion_mean_affinity[t, b] = deletion_mean[t, b]
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

function _token_and_mol_type_from_comp_id(comp_id::AbstractString, is_het::Bool; include_nonpolymer::Bool=false)
    c = uppercase(strip(String(comp_id)))
    if c in canonical_tokens || c == "UNK"
        return c == "UNK" ? "UNK" : c, chain_type_ids["PROTEIN"]
    elseif c in ("DA", "DG", "DC", "DT", "DN")
        return c, chain_type_ids["DNA"]
    elseif c in ("A", "G", "C", "U", "N")
        return c, chain_type_ids["RNA"]
    elseif !is_het
        return "UNK", chain_type_ids["PROTEIN"]
    elseif include_nonpolymer
        return "UNK", chain_type_ids["NONPOLYMER"]
    else
        return nothing, -1
    end
end

function _safeparse_int(x::AbstractString, default::Int=0)
    s = strip(String(x))
    isempty(s) && return default
    if s in (".", "?")
        return default
    end
    try
        return parse(Int, s)
    catch
        return default
    end
end

function _safeparse_f32(x::AbstractString, default::Float32=NaN32)
    s = strip(String(x))
    isempty(s) && return default
    if s in (".", "?")
        return default
    end
    try
        return Float32(parse(Float64, s))
    catch
        return default
    end
end

function _normalize_chain_id(x::AbstractString)
    s = strip(String(x))
    return isempty(s) ? "_" : s
end

function _split_cif_fields(line::AbstractString)
    out = String[]
    i = firstindex(line)
    n = lastindex(line)
    while i <= n
        while i <= n && isspace(line[i])
            i = nextind(line, i)
        end
        i > n && break
        ch = line[i]
        if ch == '\'' || ch == '"'
            q = ch
            j = nextind(line, i)
            while j <= n && line[j] != q
                j = nextind(line, j)
            end
            if j <= n
                push!(out, String(line[nextind(line, i):prevind(line, j)]))
                i = nextind(line, j)
            else
                push!(out, String(line[nextind(line, i):end]))
                break
            end
        else
            j = i
            while j <= n && !isspace(line[j])
                j = nextind(line, j)
            end
            push!(out, String(line[i:prevind(line, j)]))
            i = j
        end
    end
    return out
end

function _read_cif_loop_rows(lines::Vector{String}, start_row::Int, cols::Vector{String})
    rows = Vector{Vector{String}}()
    n = length(lines)
    row = start_row
    buffered = String[]
    ncols = length(cols)
    ncols == 0 && return rows, row
    while row <= n
        s = strip(lines[row])
        if isempty(s) || s == "#" || s == "loop_" || startswith(s, "data_") || startswith(s, "_")
            break
        end
        append!(buffered, _split_cif_fields(lines[row]))
        while length(buffered) >= ncols
            push!(rows, copy(buffered[1:ncols]))
            buffered = buffered[(ncols + 1):end]
        end
        row += 1
    end
    return rows, row
end

_identity_oper() = (
    m=(1f0, 0f0, 0f0, 0f0, 1f0, 0f0, 0f0, 0f0, 1f0),
    v=(0f0, 0f0, 0f0),
)

function _matmul3(
    a::NTuple{9, Float32},
    b::NTuple{9, Float32},
)::NTuple{9, Float32}
    return (
        a[1] * b[1] + a[2] * b[4] + a[3] * b[7],
        a[1] * b[2] + a[2] * b[5] + a[3] * b[8],
        a[1] * b[3] + a[2] * b[6] + a[3] * b[9],
        a[4] * b[1] + a[5] * b[4] + a[6] * b[7],
        a[4] * b[2] + a[5] * b[5] + a[6] * b[8],
        a[4] * b[3] + a[5] * b[6] + a[6] * b[9],
        a[7] * b[1] + a[8] * b[4] + a[9] * b[7],
        a[7] * b[2] + a[8] * b[5] + a[9] * b[8],
        a[7] * b[3] + a[8] * b[6] + a[9] * b[9],
    )
end

function _matvec3(
    m::NTuple{9, Float32},
    v::NTuple{3, Float32},
)::NTuple{3, Float32}
    return (
        m[1] * v[1] + m[2] * v[2] + m[3] * v[3],
        m[4] * v[1] + m[5] * v[2] + m[6] * v[3],
        m[7] * v[1] + m[8] * v[2] + m[9] * v[3],
    )
end

_vecadd3(a::NTuple{3, Float32}, b::NTuple{3, Float32}) = (a[1] + b[1], a[2] + b[2], a[3] + b[3])

function _compose_oper(op1, op2)
    # Compose transforms for x'' = op2(op1(x))
    return (
        m=_matmul3(op2.m, op1.m),
        v=_vecadd3(_matvec3(op2.m, op1.v), op2.v),
    )
end

_apply_oper(op, xyz::NTuple{3, Float32}) = _vecadd3(_matvec3(op.m, xyz), op.v)

function _expand_oper_token(token::AbstractString)
    s = strip(String(token))
    isempty(s) && return String[]
    m_int = match(r"^(-?\d+)-(-?\d+)$", s)
    if m_int !== nothing
        a = parse(Int, m_int.captures[1])
        b = parse(Int, m_int.captures[2])
        step = a <= b ? 1 : -1
        return [string(i) for i in a:step:b]
    end
    m_pref = match(r"^([A-Za-z_]+)(-?\d+)-([A-Za-z_]+)(-?\d+)$", s)
    if m_pref !== nothing && m_pref.captures[1] == m_pref.captures[3]
        pref = m_pref.captures[1]
        a = parse(Int, m_pref.captures[2])
        b = parse(Int, m_pref.captures[4])
        step = a <= b ? 1 : -1
        return [string(pref, i) for i in a:step:b]
    end
    return [s]
end

function _expand_oper_expression(expr::AbstractString)
    s = replace(strip(String(expr)), " " => "")
    isempty(s) && return Vector{Vector{String}}()
    groups = String[]
    if occursin("(", s)
        i = firstindex(s)
        n = lastindex(s)
        while i <= n
            if s[i] == '('
                j = findnext(==(')'), s, nextind(s, i))
                j === nothing && break
                push!(groups, String(s[nextind(s, i):prevind(s, j)]))
                i = nextind(s, j)
            else
                i = nextind(s, i)
            end
        end
        isempty(groups) && push!(groups, s)
    else
        push!(groups, s)
    end

    combos = [String[]]
    for group in groups
        ops = String[]
        for part in split(group, ',')
            append!(ops, _expand_oper_token(part))
        end
        isempty(ops) && continue
        next_combos = Vector{Vector{String}}()
        for base in combos, op in ops
            c = copy(base)
            push!(c, op)
            push!(next_combos, c)
        end
        combos = next_combos
    end
    return combos
end

function _parse_structure_records(path::AbstractString; use_assembly::Bool=false)
    p = lowercase(path)
    if endswith(p, ".pdb")
        keys = Tuple{String, Int, String, String}[]
        rec_atoms = Dict{Tuple{String, Int, String, String}, Vector{String}}()
        rec_coords = Dict{Tuple{String, Int, String, String}, Dict{String, NTuple{3, Float32}}}()
        rec_het = Dict{Tuple{String, Int, String, String}, Bool}()
        lines = readlines(path)

        pdb_op_rows = Dict{Int, Dict{Int, NTuple{4, Float32}}}()
        pdb_op_chains = Dict{Int, Set{String}}()
        active_remarks = false
        first_biomolecule = nothing
        active_chains = Set{String}()
        if use_assembly
            for line in lines
                startswith(line, "REMARK 350") || continue
                if occursin("BIOMOLECULE:", line)
                    bits = split(line, ':'; limit=2)
                    if length(bits) == 2
                        bid = _safeparse_int(strip(bits[2]), 0)
                        if first_biomolecule === nothing
                            first_biomolecule = bid
                        end
                        active_remarks = bid == first_biomolecule
                    else
                        active_remarks = false
                    end
                    empty!(active_chains)
                    continue
                end
                active_remarks || continue
                if occursin("APPLY THE FOLLOWING TO CHAINS:", line) || occursin("AND CHAINS:", line)
                    bits = split(line, ':', limit=2)
                    if length(bits) == 2
                        for item in split(bits[2], ',')
                            c = _normalize_chain_id(strip(item))
                            c == "_" && continue
                            push!(active_chains, c)
                        end
                    end
                    continue
                end
                occursin("BIOMT", line) || continue
                toks = split(strip(line))
                biomt_idx = findfirst(t -> startswith(t, "BIOMT"), toks)
                biomt_idx === nothing && continue
                (biomt_idx + 5 <= length(toks)) || continue
                biomt_tok = toks[biomt_idx]
                row_id = _safeparse_int(replace(biomt_tok, "BIOMT" => ""), 0)
                row_id in (1, 2, 3) || continue
                op_id = _safeparse_int(toks[biomt_idx + 1], 0)
                op_id > 0 || continue
                r1 = _safeparse_f32(toks[biomt_idx + 2], 0f0)
                r2 = _safeparse_f32(toks[biomt_idx + 3], 0f0)
                r3 = _safeparse_f32(toks[biomt_idx + 4], 0f0)
                rv = _safeparse_f32(toks[biomt_idx + 5], 0f0)
                if !haskey(pdb_op_rows, op_id)
                    pdb_op_rows[op_id] = Dict{Int, NTuple{4, Float32}}()
                end
                pdb_op_rows[op_id][row_id] = (r1, r2, r3, rv)
                if !haskey(pdb_op_chains, op_id)
                    pdb_op_chains[op_id] = Set{String}()
                end
                for c in active_chains
                    push!(pdb_op_chains[op_id], c)
                end
            end
        end

        for line in lines
            length(line) < 54 && continue
            rec = line[1:6]
            if !(rec == "ATOM  " || rec == "HETATM")
                continue
            end
            alt = line[17]
            if !(alt == ' ' || alt == 'A' || alt == '1')
                continue
            end
            atom_name = uppercase(strip(line[13:16]))
            isempty(atom_name) && continue
            comp_id = uppercase(strip(line[18:20]))
            chain = _normalize_chain_id(line[22:22])
            res_seq = _safeparse_int(line[23:26], 0)
            ins = strip(line[27:27])
            x = _safeparse_f32(line[31:38], NaN32)
            y = _safeparse_f32(line[39:46], NaN32)
            z = _safeparse_f32(line[47:54], NaN32)
            isfinite(x) && isfinite(y) && isfinite(z) || continue

            key = (chain, res_seq, ins, comp_id)
            if !haskey(rec_atoms, key)
                rec_atoms[key] = String[]
                rec_coords[key] = Dict{String, NTuple{3, Float32}}()
                rec_het[key] = rec == "HETATM"
                push!(keys, key)
            end
            if !haskey(rec_coords[key], atom_name)
                push!(rec_atoms[key], atom_name)
                rec_coords[key][atom_name] = (x, y, z)
            end
        end

        if use_assembly
            pdb_ops = Dict{Int, NamedTuple{(:m, :v), Tuple{NTuple{9, Float32}, NTuple{3, Float32}}}}()
            for (op_id, rows) in pdb_op_rows
                if haskey(rows, 1) && haskey(rows, 2) && haskey(rows, 3)
                    r1 = rows[1]
                    r2 = rows[2]
                    r3 = rows[3]
                    pdb_ops[op_id] = (
                        m=(r1[1], r1[2], r1[3], r2[1], r2[2], r2[3], r3[1], r3[2], r3[3]),
                        v=(r1[4], r2[4], r3[4]),
                    )
                end
            end

            if !isempty(pdb_ops)
                old_keys = copy(keys)
                old_atoms = rec_atoms
                old_coords = rec_coords
                old_het = rec_het
                all_chains = Set(k[1] for k in old_keys)
                keys = Tuple{String, Int, String, String}[]
                rec_atoms = Dict{Tuple{String, Int, String, String}, Vector{String}}()
                rec_coords = Dict{Tuple{String, Int, String, String}, Dict{String, NTuple{3, Float32}}}()
                rec_het = Dict{Tuple{String, Int, String, String}, Bool}()

                op_ids = sort!(collect(Base.keys(pdb_ops)))
                for (copy_idx, op_id) in enumerate(op_ids)
                    op = pdb_ops[op_id]
                    chains_to_apply = get(pdb_op_chains, op_id, all_chains)
                    isempty(chains_to_apply) && (chains_to_apply = all_chains)
                    for key in old_keys
                        chain, res_seq, ins, comp_id = key
                        chain in chains_to_apply || continue
                        new_chain = string(chain, copy_idx)
                        new_key = (new_chain, res_seq, ins, comp_id)
                        atoms = old_atoms[key]
                        coords_new = Dict{String, NTuple{3, Float32}}()
                        for atom_name in atoms
                            coords_new[atom_name] = _apply_oper(op, old_coords[key][atom_name])
                        end
                        push!(keys, new_key)
                        rec_atoms[new_key] = copy(atoms)
                        rec_coords[new_key] = coords_new
                        rec_het[new_key] = old_het[key]
                    end
                end
            else
                @warn "use_assembly=true requested for PDB path $(path), but no valid REMARK 350 BIOMT transforms were found; using asymmetric unit records."
            end
        end
        return keys, rec_atoms, rec_coords, rec_het
    elseif endswith(p, ".cif") || endswith(p, ".mmcif")
        keys = Tuple{String, Int, String, String}[]
        rec_atoms = Dict{Tuple{String, Int, String, String}, Vector{String}}()
        rec_coords = Dict{Tuple{String, Int, String, String}, Dict{String, NTuple{3, Float32}}}()
        rec_het = Dict{Tuple{String, Int, String, String}, Bool}()
        assembly_rows = Tuple{String, String, String}[]
        oper_map = Dict{String, NamedTuple{(:m, :v), Tuple{NTuple{9, Float32}, NTuple{3, Float32}}}}()
        assembly_scalar = Dict{String, String}()

        lines = readlines(path)
        n = length(lines)
        i = 1
        while i <= n
            line = strip(lines[i])
            if startswith(line, "_pdbx_struct_assembly_gen.")
                fields = _split_cif_fields(lines[i])
                if length(fields) >= 2
                    assembly_scalar[fields[1]] = fields[2]
                end
                i += 1
                continue
            elseif line == "loop_"
                cols = String[]
                j = i + 1
                while j <= n && startswith(strip(lines[j]), "_")
                    push!(cols, strip(lines[j]))
                    j += 1
                end
                rows, row_after = _read_cif_loop_rows(lines, j, cols)
                if any(startswith(c, "_atom_site.") for c in cols)
                    col_idx = Dict{String, Int}(c => k for (k, c) in enumerate(cols))
                    col = (fields, names::Vector{String}, default::String="") -> begin
                        for name in names
                            if haskey(col_idx, name)
                                return fields[col_idx[name]]
                            end
                        end
                        return default
                    end
                    for fields in rows
                        group = uppercase(strip(col(fields, ["_atom_site.group_PDB"])))
                        if !(group == "ATOM" || group == "HETATM")
                            continue
                        end
                        model_num = _safeparse_int(col(fields, ["_atom_site.pdbx_PDB_model_num"]), 1)
                        model_num == 1 || continue
                        alt = strip(col(fields, ["_atom_site.label_alt_id"], "."))
                        if !(alt == "." || alt == "?" || alt == "A" || alt == "1")
                            continue
                        end
                        atom_name = uppercase(strip(col(fields, ["_atom_site.label_atom_id", "_atom_site.auth_atom_id"])))
                        comp_id = uppercase(strip(col(fields, ["_atom_site.label_comp_id", "_atom_site.auth_comp_id"])))
                        chain = _normalize_chain_id(col(fields, ["_atom_site.label_asym_id", "_atom_site.auth_asym_id"], "_"))
                        seq_raw = col(fields, ["_atom_site.label_seq_id", "_atom_site.auth_seq_id"], "0")
                        res_seq = _safeparse_int(seq_raw, 0)
                        ins = strip(col(fields, ["_atom_site.pdbx_PDB_ins_code"], ""))
                        x = _safeparse_f32(col(fields, ["_atom_site.Cartn_x"]), NaN32)
                        y = _safeparse_f32(col(fields, ["_atom_site.Cartn_y"]), NaN32)
                        z = _safeparse_f32(col(fields, ["_atom_site.Cartn_z"]), NaN32)
                        if isempty(atom_name) || isempty(comp_id) || !isfinite(x) || !isfinite(y) || !isfinite(z)
                            continue
                        end

                        key = (chain, res_seq, ins, comp_id)
                        if !haskey(rec_atoms, key)
                            rec_atoms[key] = String[]
                            rec_coords[key] = Dict{String, NTuple{3, Float32}}()
                            rec_het[key] = group == "HETATM"
                            push!(keys, key)
                        end
                        if !haskey(rec_coords[key], atom_name)
                            push!(rec_atoms[key], atom_name)
                            rec_coords[key][atom_name] = (x, y, z)
                        end
                    end
                    i = row_after
                    continue
                elseif any(startswith(c, "_pdbx_struct_oper_list.") for c in cols)
                    col_idx = Dict{String, Int}(c => k for (k, c) in enumerate(cols))
                    col = (fields, names::Vector{String}, default::String="") -> begin
                        for name in names
                            if haskey(col_idx, name)
                                return fields[col_idx[name]]
                            end
                        end
                        return default
                    end
                    for fields in rows
                        op_id = strip(col(fields, ["_pdbx_struct_oper_list.id"]))
                        isempty(op_id) && continue
                        m11 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[1][1]"]), 1f0)
                        m12 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[1][2]"]), 0f0)
                        m13 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[1][3]"]), 0f0)
                        m21 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[2][1]"]), 0f0)
                        m22 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[2][2]"]), 1f0)
                        m23 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[2][3]"]), 0f0)
                        m31 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[3][1]"]), 0f0)
                        m32 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[3][2]"]), 0f0)
                        m33 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.matrix[3][3]"]), 1f0)
                        v1 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.vector[1]"]), 0f0)
                        v2 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.vector[2]"]), 0f0)
                        v3 = _safeparse_f32(col(fields, ["_pdbx_struct_oper_list.vector[3]"]), 0f0)
                        oper_map[op_id] = (
                            m=(m11, m12, m13, m21, m22, m23, m31, m32, m33),
                            v=(v1, v2, v3),
                        )
                    end
                    i = row_after
                    continue
                elseif any(startswith(c, "_pdbx_struct_assembly_gen.") for c in cols)
                    col_idx = Dict{String, Int}(c => k for (k, c) in enumerate(cols))
                    col = (fields, names::Vector{String}, default::String="") -> begin
                        for name in names
                            if haskey(col_idx, name)
                                return fields[col_idx[name]]
                            end
                        end
                        return default
                    end
                    for fields in rows
                        aid = strip(col(fields, ["_pdbx_struct_assembly_gen.assembly_id"]))
                        op_expr = strip(col(fields, ["_pdbx_struct_assembly_gen.oper_expression"]))
                        asym_list = strip(col(fields, ["_pdbx_struct_assembly_gen.asym_id_list"]))
                        if isempty(aid) || isempty(op_expr) || isempty(asym_list)
                            continue
                        end
                        push!(assembly_rows, (aid, op_expr, asym_list))
                    end
                    i = row_after
                    continue
                end
                i = row_after
                continue
            end
            i += 1
        end

        if isempty(assembly_rows) &&
           haskey(assembly_scalar, "_pdbx_struct_assembly_gen.assembly_id") &&
           haskey(assembly_scalar, "_pdbx_struct_assembly_gen.oper_expression") &&
           haskey(assembly_scalar, "_pdbx_struct_assembly_gen.asym_id_list")
            push!(
                assembly_rows,
                (
                    assembly_scalar["_pdbx_struct_assembly_gen.assembly_id"],
                    assembly_scalar["_pdbx_struct_assembly_gen.oper_expression"],
                    assembly_scalar["_pdbx_struct_assembly_gen.asym_id_list"],
                ),
            )
        end

        if use_assembly && !isempty(assembly_rows) && !isempty(oper_map)
            selected_assembly = assembly_rows[1][1]
            copy_idx_by_combo = Dict{String, Int}()
            copy_op = Dict{Int, NamedTuple{(:m, :v), Tuple{NTuple{9, Float32}, NTuple{3, Float32}}}}()
            copy_asym = Dict{Int, Set{String}}()
            next_copy = 1

            for (aid, op_expr, asym_expr) in assembly_rows
                aid == selected_assembly || continue
                combos = _expand_oper_expression(op_expr)
                isempty(combos) && continue
                asym_ids = [_normalize_chain_id(s) for s in split(asym_expr, ",") if !isempty(strip(s))]
                isempty(asym_ids) && continue
                for combo in combos
                    combo_key = join(combo, "*")
                    if !haskey(copy_idx_by_combo, combo_key)
                        op = _identity_oper()
                        valid = true
                        for op_id in combo
                            if !haskey(oper_map, op_id)
                                valid = false
                                break
                            end
                            op = _compose_oper(op, oper_map[op_id])
                        end
                        valid || continue
                        copy_idx_by_combo[combo_key] = next_copy
                        copy_op[next_copy] = op
                        copy_asym[next_copy] = Set{String}()
                        next_copy += 1
                    end
                    copy_idx = copy_idx_by_combo[combo_key]
                    for asym in asym_ids
                        push!(copy_asym[copy_idx], asym)
                    end
                end
            end

            if !isempty(copy_op)
                old_keys = copy(keys)
                old_atoms = rec_atoms
                old_coords = rec_coords
                old_het = rec_het
                keys = Tuple{String, Int, String, String}[]
                rec_atoms = Dict{Tuple{String, Int, String, String}, Vector{String}}()
                rec_coords = Dict{Tuple{String, Int, String, String}, Dict{String, NTuple{3, Float32}}}()
                rec_het = Dict{Tuple{String, Int, String, String}, Bool}()

                copy_ids = sort!(collect(Base.keys(copy_op)))
                for copy_idx in copy_ids
                    asym_set = copy_asym[copy_idx]
                    op = copy_op[copy_idx]
                    for key in old_keys
                        chain, res_seq, ins, comp_id = key
                        chain in asym_set || continue
                        new_chain = string(chain, copy_idx)
                        new_key = (new_chain, res_seq, ins, comp_id)
                        coords_new = Dict{String, NTuple{3, Float32}}()
                        atoms = old_atoms[key]
                        for atom_name in atoms
                            coords_new[atom_name] = _apply_oper(op, old_coords[key][atom_name])
                        end
                        push!(keys, new_key)
                        rec_atoms[new_key] = copy(atoms)
                        rec_coords[new_key] = coords_new
                        rec_het[new_key] = old_het[key]
                    end
                end
            end
        end
        return keys, rec_atoms, rec_coords, rec_het
    else
        error("Unsupported structure format (expected .pdb/.cif/.mmcif): $path")
    end
end

function load_structure_tokens(
    path::AbstractString;
    include_chains::Union{Nothing, Vector{String}}=nothing,
    include_nonpolymer::Bool=false,
    use_assembly::Bool=false,
)
    keys, rec_atoms, rec_coords, rec_het = _parse_structure_records(path; use_assembly=use_assembly)

    keep_chain = nothing
    if include_chains !== nothing
        keep_chain = Set(_normalize_chain_id(c) for c in include_chains)
    end

    residue_tokens = String[]
    mol_types = Int[]
    chain_labels = String[]
    residue_indices = Int[]
    token_atom_names = Vector{Vector{String}}()
    token_atom_coords = Vector{Dict{String, NTuple{3, Float32}}}()

    for key in keys
        chain, res_seq, _, comp_id = key
        if keep_chain !== nothing && !(chain in keep_chain)
            continue
        end
        tok, mt = _token_and_mol_type_from_comp_id(comp_id, rec_het[key]; include_nonpolymer=include_nonpolymer)
        tok === nothing && continue

        push!(residue_tokens, tok)
        push!(mol_types, mt)
        push!(chain_labels, chain)
        push!(residue_indices, res_seq)
        push!(token_atom_names, copy(rec_atoms[key]))
        push!(token_atom_coords, copy(rec_coords[key]))
    end

    isempty(residue_tokens) && error("No polymer residues parsed from structure: $path")

    chain_to_asym = Dict{String, Int}()
    next_asym = 0
    asym_ids = Int[]
    for c in chain_labels
        if !haskey(chain_to_asym, c)
            chain_to_asym[c] = next_asym
            next_asym += 1
        end
        push!(asym_ids, chain_to_asym[c])
    end
    entity_ids = copy(asym_ids)
    sym_ids = zeros(Int, length(asym_ids))

    return (
        residue_tokens=residue_tokens,
        mol_types=mol_types,
        asym_ids=asym_ids,
        entity_ids=entity_ids,
        sym_ids=sym_ids,
        residue_indices=residue_indices,
        token_atom_names=token_atom_names,
        token_atom_coords=token_atom_coords,
        chain_labels=chain_labels,
    )
end

function build_design_features_from_structure(
    path::AbstractString;
    include_chains::Union{Nothing, Vector{String}}=nothing,
    include_nonpolymer::Bool=false,
    design_mask::Union{Nothing, AbstractVector{Bool}}=nothing,
    chain_design_mask::Union{Nothing, AbstractVector{Bool}}=nothing,
    binding_type::Union{Nothing, Vector{Int}}=nothing,
    ss_type::Union{Nothing, Vector{Int}}=nothing,
    structure_group::Union{Nothing, Vector{Int}}=nothing,
    target_msa_mask::Union{Nothing, AbstractVector{Bool}}=nothing,
    msa_sequences::Union{Nothing,Vector{String}}=nothing,
    msa_paired_rows::Union{Nothing,AbstractVector{Bool}}=nothing,
    msa_mask_rows::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    msa_has_deletion_rows::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    msa_deletion_value_rows::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    max_msa_rows::Union{Nothing,Int}=nothing,
    affinity_token_mask::Union{Nothing, AbstractVector{Bool}}=nothing,
    affinity_mw::Float32=0f0,
    template_paths::Vector{String}=String[],
    max_templates::Union{Nothing,Int}=nothing,
    template_include_chains::Union{Nothing,Vector{String}}=nothing,
    bonds::Vector{NTuple{3, Int}}=NTuple{3, Int}[],
    contact_pairs::Vector{NTuple{3, Any}}=NTuple{3, Any}[],
    batch::Int=1,
    pad_atom_multiple::Int=32,
    force_atom14_for_designed_protein::Bool=true,
)
    parsed = load_structure_tokens(path; include_chains=include_chains, include_nonpolymer=include_nonpolymer)
    T = length(parsed.residue_tokens)
    design_v = design_mask === nothing ? falses(T) : collect(design_mask)
    groups_v = structure_group === nothing ? ones(Int, T) : structure_group
    return build_design_features(
        parsed.residue_tokens;
        mol_types=parsed.mol_types,
        asym_ids=parsed.asym_ids,
        entity_ids=parsed.entity_ids,
        sym_ids=parsed.sym_ids,
        residue_indices=parsed.residue_indices,
        design_mask=design_v,
        chain_design_mask=chain_design_mask,
        binding_type=binding_type,
        ss_type=ss_type,
        structure_group=groups_v,
        target_msa_mask=target_msa_mask,
        msa_sequences=msa_sequences,
        msa_paired_rows=msa_paired_rows,
        msa_mask_rows=msa_mask_rows,
        msa_has_deletion_rows=msa_has_deletion_rows,
        msa_deletion_value_rows=msa_deletion_value_rows,
        max_msa_rows=max_msa_rows,
        affinity_token_mask=affinity_token_mask,
        affinity_mw=affinity_mw,
        template_paths=template_paths,
        max_templates=max_templates,
        template_include_chains=template_include_chains,
        bonds=bonds,
        contact_pairs=contact_pairs,
        batch=batch,
        pad_atom_multiple=pad_atom_multiple,
        force_atom14_for_designed_protein=force_atom14_for_designed_protein,
        token_atom_names_override=parsed.token_atom_names,
        token_atom_coords_override=parsed.token_atom_coords,
    )
end
