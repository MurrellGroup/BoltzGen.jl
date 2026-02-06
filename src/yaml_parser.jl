function _ydict_get(d, key::AbstractString, default=nothing)
    if !(d isa AbstractDict)
        return default
    end
    if haskey(d, key)
        return d[key]
    end
    sk = Symbol(key)
    if haskey(d, sk)
        return d[sk]
    end
    return default
end

function _as_list(x)
    if x === nothing
        return Any[]
    elseif x isa AbstractVector
        return collect(x)
    else
        return Any[x]
    end
end

function _string(x)
    return string(x)
end

function _upper(x)
    return uppercase(strip(string(x)))
end

function _as_boolish(x, default::Bool=false)
    if x === nothing
        return default
    elseif x isa Bool
        return x
    elseif x isa Integer
        return Int(x) != 0
    end
    s = lowercase(strip(string(x)))
    if s in ("1", "true", "t", "yes", "y", "on")
        return true
    elseif s in ("0", "false", "f", "no", "n", "off", "")
        return false
    end
    return default
end

function _parse_index_spec(spec, chain_len::Int)
    if spec === nothing
        return collect(1:chain_len)
    end
    s = strip(string(spec))
    isempty(s) && return collect(1:chain_len)
    lowercase(s) == "all" && return collect(1:chain_len)

    out = Int[]
    for part0 in split(s, ',')
        part = strip(part0)
        isempty(part) && continue
        if occursin("..", part)
            if startswith(part, "..")
                hi = parse(Int, strip(part[3:end]))
                lo = 1
            elseif endswith(part, "..")
                lo = parse(Int, strip(part[1:end-2]))
                hi = chain_len
            else
                bits = split(part, "..")
                length(bits) == 2 || continue
                lo = parse(Int, strip(bits[1]))
                hi = parse(Int, strip(bits[2]))
            end
            lo = clamp(lo, 1, chain_len)
            hi = clamp(hi, 1, chain_len)
            if lo <= hi
                append!(out, lo:hi)
            else
                append!(out, hi:lo)
            end
        else
            idx = parse(Int, part)
            if 1 <= idx <= chain_len
                push!(out, idx)
            end
        end
    end

    seen = Set{Int}()
    uniq = Int[]
    for idx in out
        if !(idx in seen)
            push!(uniq, idx)
            push!(seen, idx)
        end
    end
    return uniq
end

function _parse_count_spec(spec)
    s = strip(string(spec))
    if occursin("..", s)
        bits = split(s, "..")
        length(bits) == 2 || error("Invalid count spec: $s")
        lo = parse(Int, strip(bits[1]))
        hi = parse(Int, strip(bits[2]))
        lo <= hi || ((lo, hi) = (hi, lo))
        return collect(lo:hi)
    end
    n = parse(Int, s)
    return [n]
end

function _default_design_token(chain_type::AbstractString)
    ct = uppercase(chain_type)
    if ct == "PROTEIN"
        return "GLY"
    elseif ct == "DNA"
        return "DN"
    elseif ct == "RNA"
        return "N"
    end
    return "UNK"
end

function _letter_to_token(chain_type::AbstractString, c::Char)
    ct = uppercase(chain_type)
    if ct == "PROTEIN"
        return get(prot_letter_to_token, uppercase(c), "UNK")
    elseif ct == "DNA"
        return get(dna_letter_to_token, uppercase(c), "DN")
    elseif ct == "RNA"
        return get(rna_letter_to_token, uppercase(c), "N")
    end
    return "UNK"
end

function _parse_polymer_sequence(raw::AbstractString, chain_type::AbstractString, rng::AbstractRNG)
    tokens = String[]
    design_mask = Bool[]

    for part0 in split(string(raw), ',')
        part = strip(part0)
        isempty(part) && continue
        for m in eachmatch(r"\d+\.\.\d+|\d+|[A-Za-z]", part)
            tok = string(m.match)
            if occursin("..", tok)
                bits = split(tok, "..")
                lo = parse(Int, bits[1])
                hi = parse(Int, bits[2])
                lo <= hi || ((lo, hi) = (hi, lo))
                n = rand(rng, lo:hi)
                append!(tokens, fill(_default_design_token(chain_type), n))
                append!(design_mask, fill(true, n))
            elseif all(isdigit, tok)
                n = parse(Int, tok)
                append!(tokens, fill(_default_design_token(chain_type), n))
                append!(design_mask, fill(true, n))
            else
                push!(tokens, _letter_to_token(chain_type, tok[1]))
                push!(design_mask, false)
            end
        end
    end

    return tokens, design_mask
end

function _parse_binding_spec(spec, n::Int)
    out = fill(binding_type_ids["UNSPECIFIED"], n)
    if spec === nothing
        return out
    elseif spec isa AbstractString
        s = strip(string(spec))
        for i in 1:min(n, length(s))
            c = lowercase(s[i])
            if c == 'b'
                out[i] = binding_type_ids["BINDING"]
            elseif c == 'n'
                out[i] = binding_type_ids["NOT_BINDING"]
            else
                out[i] = binding_type_ids["UNSPECIFIED"]
            end
        end
        return out
    elseif spec isa AbstractDict
        if haskey(spec, "binding") || haskey(spec, :binding)
            idxs = _parse_index_spec(_ydict_get(spec, "binding", nothing), n)
            out[idxs] .= binding_type_ids["BINDING"]
        end
        if haskey(spec, "not_binding") || haskey(spec, :not_binding)
            idxs = _parse_index_spec(_ydict_get(spec, "not_binding", nothing), n)
            out[idxs] .= binding_type_ids["NOT_BINDING"]
        end
        return out
    end
    return out
end

function _parse_ss_spec(spec, n::Int)
    out = fill(ss_type_ids["UNSPECIFIED"], n)
    if spec === nothing
        return out
    elseif spec isa AbstractString
        s = strip(string(spec))
        for i in 1:min(n, length(s))
            c = lowercase(s[i])
            if c == 'l'
                out[i] = ss_type_ids["LOOP"]
            elseif c == 'h'
                out[i] = ss_type_ids["HELIX"]
            elseif c == 's'
                out[i] = ss_type_ids["SHEET"]
            else
                out[i] = ss_type_ids["UNSPECIFIED"]
            end
        end
        return out
    elseif spec isa AbstractDict
        if haskey(spec, "loop") || haskey(spec, :loop)
            idxs = _parse_index_spec(_ydict_get(spec, "loop", nothing), n)
            out[idxs] .= ss_type_ids["LOOP"]
        end
        if haskey(spec, "helix") || haskey(spec, :helix)
            idxs = _parse_index_spec(_ydict_get(spec, "helix", nothing), n)
            out[idxs] .= ss_type_ids["HELIX"]
        end
        if haskey(spec, "sheet") || haskey(spec, :sheet)
            idxs = _parse_index_spec(_ydict_get(spec, "sheet", nothing), n)
            out[idxs] .= ss_type_ids["SHEET"]
        end
        return out
    end
    return out
end

function _token_center(tok::String, mol_type_id::Int, atom_names::Vector{String}, atom_coords::Dict{String,NTuple{3,Float32}})
    if isempty(atom_names)
        return (0f0, 0f0, 0f0)
    end

    preferred = if mol_type_id == chain_type_ids["PROTEIN"]
        "CA"
    elseif mol_type_id == chain_type_ids["DNA"] || mol_type_id == chain_type_ids["RNA"]
        "C1'"
    else
        atom_names[1]
    end

    if haskey(atom_coords, preferred)
        return atom_coords[preferred]
    end
    if haskey(res_to_center_atom, tok)
        c = res_to_center_atom[tok]
        if haskey(atom_coords, c)
            return atom_coords[c]
        end
    end

    sx = 0f0
    sy = 0f0
    sz = 0f0
    n = 0
    for name in atom_names
        if haskey(atom_coords, name)
            xyz = atom_coords[name]
            sx += xyz[1]
            sy += xyz[2]
            sz += xyz[3]
            n += 1
        end
    end
    if n == 0
        return (0f0, 0f0, 0f0)
    end
    invn = 1f0 / Float32(n)
    return (sx * invn, sy * invn, sz * invn)
end

function _resolve_yaml_path(path_raw, base_dir::AbstractString, rng::AbstractRNG)
    if path_raw isa AbstractVector
        isempty(path_raw) && error("Empty path list in YAML")
        path_raw = rand(rng, collect(path_raw))
    end
    p = string(path_raw)
    if isabspath(p)
        return normpath(p)
    end
    return normpath(joinpath(base_dir, p))
end

function _resolve_file_entity_spec(spec::AbstractDict, base_dir::AbstractString, rng::AbstractRNG)
    current_spec = spec
    current_base = base_dir

    while true
        path_raw = _ydict_get(current_spec, "path", nothing)
        path = _resolve_yaml_path(path_raw, current_base, rng)
        lower = lowercase(path)
        if endswith(lower, ".yaml") || endswith(lower, ".yml")
            nested = YAML.load_file(path)
            nested isa AbstractDict || error("Nested file spec must be a YAML mapping: $(path)")
            current_spec = nested
            current_base = dirname(path)
            continue
        end
        return current_spec, path
    end
end

function _insert_token!(
    idx::Int,
    tok::String,
    mt::Int,
    asym::Int,
    ent::Int,
    sym::Int,
    chain::String,
    res_idx::Int,
    include_mask::AbstractVector{Bool},
    residue_tokens::Vector{String},
    mol_types::Vector{Int},
    asym_ids::Vector{Int},
    entity_ids::Vector{Int},
    sym_ids::Vector{Int},
    residue_indices::Vector{Int},
    chain_labels::Vector{String},
    token_atom_names::Vector{Vector{String}},
    token_atom_coords::Vector{Dict{String,NTuple{3,Float32}}},
    design_mask::AbstractVector{Bool},
    binding_type::Vector{Int},
    ss_type::Vector{Int},
    structure_group::Vector{Int},
    target_msa_mask::Vector{Bool},
    cyclic_period::Vector{Int},
    ss_insert_id::Int,
    target_msa::Bool=false,
    cyclic_p::Int=0,
)
    insert!(residue_tokens, idx, tok)
    insert!(mol_types, idx, mt)
    insert!(asym_ids, idx, asym)
    insert!(entity_ids, idx, ent)
    insert!(sym_ids, idx, sym)
    insert!(residue_indices, idx, res_idx)
    insert!(chain_labels, idx, chain)
    insert!(token_atom_names, idx, String[])
    insert!(token_atom_coords, idx, Dict{String,NTuple{3,Float32}}())

    insert!(include_mask, idx, true)
    insert!(design_mask, idx, true)
    insert!(binding_type, idx, binding_type_ids["UNSPECIFIED"])
    insert!(ss_type, idx, ss_insert_id)
    insert!(structure_group, idx, 0)
    insert!(target_msa_mask, idx, target_msa)
    insert!(cyclic_period, idx, cyclic_p)
end

function _parse_file_entity(spec, base_dir::AbstractString, include_nonpolymer::Bool, rng::AbstractRNG)
    spec, path = _resolve_file_entity_spec(spec, base_dir, rng)
    use_assembly = _as_boolish(_ydict_get(spec, "use_assembly", false), false)
    parsed = load_structure_tokens(path; include_nonpolymer=include_nonpolymer, use_assembly=use_assembly)

    residue_tokens = copy(parsed.residue_tokens)
    mol_types = copy(parsed.mol_types)
    asym_ids = copy(parsed.asym_ids)
    entity_ids = copy(parsed.entity_ids)
    sym_ids = copy(parsed.sym_ids)
    residue_indices = copy(parsed.residue_indices)
    chain_labels = copy(parsed.chain_labels)
    token_atom_names = copy(parsed.token_atom_names)
    token_atom_coords = copy(parsed.token_atom_coords)
    target_msa_mask = fill(_as_boolish(_ydict_get(spec, "msa", false), false), length(residue_tokens))
    cyclic_period = zeros(Int, length(residue_tokens))

    n = length(residue_tokens)
    include_mask = trues(n)

    chain_global_idxs(chain_id::AbstractString, res_spec=nothing) = begin
        idxs = findall(==(string(chain_id)), chain_labels)
        isempty(idxs) && error("Chain $(chain_id) not found in file entity: $(path)")
        if res_spec === nothing
            return idxs
        end
        local_idxs = _parse_index_spec(res_spec, length(idxs))
        return [idxs[i] for i in local_idxs]
    end

    include_spec = _ydict_get(spec, "include", "all")
    if include_spec isa AbstractString
        lowercase(strip(string(include_spec))) == "all" || error("Unsupported include spec: $(include_spec)")
    else
        include_mask .= false
        for e in _as_list(include_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in include")
            ridx = _ydict_get(c, "res_index", nothing)
            include_mask[chain_global_idxs(string(cid), ridx)] .= true
        end
    end

    include_proximity = _ydict_get(spec, "include_proximity", nothing)
    if include_proximity !== nothing
        centers = [
            _token_center(residue_tokens[i], mol_types[i], token_atom_names[i], token_atom_coords[i])
            for i in 1:n
        ]
        prox = falses(n)
        for e in _as_list(include_proximity)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in include_proximity")
            ridx = _ydict_get(c, "res_index", nothing)
            radius = Float64(_ydict_get(c, "radius", 0.0))
            qidxs = chain_global_idxs(string(cid), ridx)
            for i in 1:n
                ci = centers[i]
                for q in qidxs
                    cq = centers[q]
                    dx = Float64(ci[1] - cq[1])
                    dy = Float64(ci[2] - cq[2])
                    dz = Float64(ci[3] - cq[3])
                    if dx * dx + dy * dy + dz * dz <= radius * radius
                        prox[i] = true
                        break
                    end
                end
            end
        end
        include_mask .&= prox
    end

    exclude_spec = _ydict_get(spec, "exclude", nothing)
    if exclude_spec !== nothing
        for e in _as_list(exclude_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in exclude")
            ridx = _ydict_get(c, "res_index", nothing)
            include_mask[chain_global_idxs(string(cid), ridx)] .= false
        end
    end

    design_mask = falses(n)
    design_spec = _ydict_get(spec, "design", nothing)
    if design_spec !== nothing
        for e in _as_list(design_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in design")
            ridx = _ydict_get(c, "res_index", nothing)
            design_mask[chain_global_idxs(string(cid), ridx)] .= true
        end
    end

    not_design_spec = _ydict_get(spec, "not_design", nothing)
    if not_design_spec !== nothing
        for e in _as_list(not_design_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in not_design")
            ridx = _ydict_get(c, "res_index", nothing)
            design_mask[chain_global_idxs(string(cid), ridx)] .= false
        end
    end

    binding_type = fill(binding_type_ids["UNSPECIFIED"], n)
    bind_spec = _ydict_get(spec, "binding_types", nothing)
    if bind_spec !== nothing
        for e in _as_list(bind_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in binding_types")
            all_idx = chain_global_idxs(string(cid), nothing)
            b = _ydict_get(c, "binding", nothing)
            nb = _ydict_get(c, "not_binding", nothing)
            if b !== nothing
                idxs = (b isa AbstractString && lowercase(strip(string(b))) == "all") ? all_idx : chain_global_idxs(string(cid), b)
                binding_type[idxs] .= binding_type_ids["BINDING"]
            end
            if nb !== nothing
                idxs = (nb isa AbstractString && lowercase(strip(string(nb))) == "all") ? all_idx : chain_global_idxs(string(cid), nb)
                binding_type[idxs] .= binding_type_ids["NOT_BINDING"]
            end
        end
    end

    ss_type = fill(ss_type_ids["UNSPECIFIED"], n)
    ss_spec = _ydict_get(spec, "secondary_structure", nothing)
    if ss_spec !== nothing
        for e in _as_list(ss_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in secondary_structure")
            all_idx = chain_global_idxs(string(cid), nothing)
            for (k, sid) in (("loop", ss_type_ids["LOOP"]), ("helix", ss_type_ids["HELIX"]), ("sheet", ss_type_ids["SHEET"]))
                v = _ydict_get(c, k, nothing)
                v === nothing && continue
                idxs = (v isa AbstractString && lowercase(strip(string(v))) == "all") ? all_idx : chain_global_idxs(string(cid), v)
                ss_type[idxs] .= sid
            end
        end
    end

    sg = _ydict_get(spec, "structure_groups", nothing)
    structure_group = if sg === nothing || (sg isa AbstractString && lowercase(strip(string(sg))) == "all") || (sg isa Integer && Int(sg) == 1)
        ones(Int, n)
    else
        g = zeros(Int, n)
        for e in _as_list(sg)
            gr = _ydict_get(e, "group", e)
            vis = Int(_ydict_get(gr, "visibility", 0))
            gid = _ydict_get(gr, "id", nothing)
            gid === nothing && error("Missing group.id in structure_groups")
            if string(gid) == "all"
                g .= vis
            else
                ridx = _ydict_get(gr, "res_index", nothing)
                g[chain_global_idxs(string(gid), ridx)] .= vis
            end
        end
        g
    end

    insert_spec = _ydict_get(spec, "design_insertions", nothing)
    if insert_spec !== nothing
        num_inserted = Dict{String, Int}()
        for e in _as_list(insert_spec)
            ins = _ydict_get(e, "insertion", e)
            cid = _ydict_get(ins, "id", nothing)
            cid === nothing && error("Missing insertion.id in design_insertions")
            chain_id = string(cid)
            res_index = Int(_ydict_get(ins, "res_index", 1))
            num_spec = _ydict_get(ins, "num_residues", nothing)
            num_spec === nothing && error("Missing insertion.num_residues")
            ss_insert = _upper(_ydict_get(ins, "secondary_structure", "UNSPECIFIED"))
            ss_insert_id = get(ss_type_ids, ss_insert, ss_type_ids["UNSPECIFIED"])

            if !haskey(num_inserted, chain_id)
                num_inserted[chain_id] = 0
            end
            nins = rand(rng, _parse_count_spec(num_spec))
            r0 = (res_index - 1) + num_inserted[chain_id]

            chain_idxs = findall(==(chain_id), chain_labels)
            isempty(chain_idxs) && error("Chain $(chain_id) not found for insertion")
            ins_pos = clamp(r0 + 1, 1, length(chain_idxs) + 1)
            gidx = if ins_pos <= length(chain_idxs)
                chain_idxs[ins_pos]
            else
                chain_idxs[end] + 1
            end

            mt = mol_types[chain_idxs[1]]
            tok = if mt == chain_type_ids["PROTEIN"]
                "GLY"
            elseif mt == chain_type_ids["DNA"]
                "DN"
            elseif mt == chain_type_ids["RNA"]
                "N"
            else
                "UNK"
            end
            asym = asym_ids[chain_idxs[1]]
            ent = entity_ids[chain_idxs[1]]
            sym = sym_ids[chain_idxs[1]]
            resv = residue_indices[chain_idxs[min(ins_pos, length(chain_idxs))]]

            for j in 0:(nins - 1)
                _insert_token!(
                    gidx + j,
                    tok,
                    mt,
                    asym,
                    ent,
                    sym,
                    chain_id,
                    resv,
                    include_mask,
                    residue_tokens,
                    mol_types,
                    asym_ids,
                    entity_ids,
                    sym_ids,
                    residue_indices,
                    chain_labels,
                    token_atom_names,
                    token_atom_coords,
                    design_mask,
                    binding_type,
                    ss_type,
                    structure_group,
                    target_msa_mask,
                    cyclic_period,
                    ss_insert_id,
                )
            end
            num_inserted[chain_id] += nins
        end
    end

    keep = findall(include_mask)
    residue_tokens = residue_tokens[keep]
    mol_types = mol_types[keep]
    asym_ids = asym_ids[keep]
    entity_ids = entity_ids[keep]
    sym_ids = sym_ids[keep]
    residue_indices = residue_indices[keep]
    chain_labels = chain_labels[keep]
    token_atom_names = token_atom_names[keep]
    token_atom_coords = token_atom_coords[keep]
    design_mask = design_mask[keep]
    binding_type = binding_type[keep]
    ss_type = ss_type[keep]
    structure_group = structure_group[keep]
    target_msa_mask = target_msa_mask[keep]
    cyclic_period = cyclic_period[keep]

    reset_spec = _ydict_get(spec, "reset_res_index", nothing)
    if reset_spec !== nothing
        for e in _as_list(reset_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in reset_res_index")
            idxs = findall(==(string(cid)), chain_labels)
            for (k, idx) in enumerate(idxs)
                residue_indices[idx] = k
            end
        end
    end

    cyc_spec = _ydict_get(spec, "add_cyclization", nothing)
    if cyc_spec !== nothing
        for e in _as_list(cyc_spec)
            c = _ydict_get(e, "chain", e)
            cid = _ydict_get(c, "id", nothing)
            cid === nothing && error("Missing chain.id in add_cyclization")
            idxs = findall(==(string(cid)), chain_labels)
            !isempty(idxs) || continue
            cyc_val = length(idxs)
            for idx in idxs
                cyclic_period[idx] = cyc_val
            end
        end
    end

    return (
        residue_tokens=residue_tokens,
        mol_types=mol_types,
        asym_ids=asym_ids,
        entity_ids=entity_ids,
        sym_ids=sym_ids,
        residue_indices=residue_indices,
        chain_labels=chain_labels,
        token_atom_names=token_atom_names,
        token_atom_coords=token_atom_coords,
        design_mask=design_mask,
        binding_type=binding_type,
        ss_type=ss_type,
        structure_group=structure_group,
        target_msa_mask=target_msa_mask,
        cyclic_period=cyclic_period,
    )
end

function _canonicalize_schema(schema)
    if schema isa AbstractDict && (_ydict_get(schema, "entities", nothing) !== nothing)
        return schema
    end

    # Support legacy single-file YAMLs without top-level entities.
    file_keys = (
        "path",
        "include",
        "exclude",
        "include_proximity",
        "structure_groups",
        "design",
        "not_design",
        "binding_types",
        "secondary_structure",
        "design_insertions",
        "reset_res_index",
        "fuse",
        "use_assembly",
        "msa",
        "add_cyclization",
        "symmetric_group",
    )
    if schema isa AbstractDict && any(_ydict_get(schema, k, nothing) !== nothing for k in file_keys)
        file_spec = Dict{Any, Any}()
        for k in file_keys
            v = _ydict_get(schema, k, nothing)
            if v !== nothing
                file_spec[k] = v
            end
        end
        out = Dict{Any, Any}()
        out["entities"] = [Dict("file" => file_spec)]
        constraints = _ydict_get(schema, "constraints", nothing)
        if constraints !== nothing
            out["constraints"] = constraints
        end
        leaving_atoms = _ydict_get(schema, "leaving_atoms", nothing)
        if leaving_atoms !== nothing
            out["leaving_atoms"] = leaving_atoms
        end
        return out
    end

    error("YAML schema must contain 'entities' or top-level file keys")
end

function _extract_total_len(schema)
    constraints = _ydict_get(schema, "constraints", nothing)
    constraints === nothing && return nothing
    for c in _as_list(constraints)
        tl = _ydict_get(c, "total_len", nothing)
        tl === nothing && continue
        minlen = Int(_ydict_get(tl, "min", 0))
        maxlen = Int(_ydict_get(tl, "max", typemax(Int)))
        return (min=minlen, max=maxlen)
    end
    return nothing
end

function _resolve_constraint_chain(name::String, chain_aliases::Dict{String, Vector{String}})
    if haskey(chain_aliases, name)
        vals = chain_aliases[name]
        if length(vals) == 1
            return vals[1]
        end
        error("Constraint chain '$name' is ambiguous after renaming: $(join(vals, ", "))")
    end
    return name
end

function parse_design_yaml(
    yaml_path::AbstractString;
    include_nonpolymer::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
    max_total_len_trials::Int=128,
)
    base_dir = dirname(abspath(yaml_path))
    raw = YAML.load_file(yaml_path)
    schema = _canonicalize_schema(raw)
    total_len = _extract_total_len(schema)

    last_err = nothing
    for _ in 1:max_total_len_trials
        try
            residue_tokens = String[]
            mol_types = Int[]
            asym_ids = Int[]
            entity_ids = Int[]
            sym_ids = Int[]
            residue_indices = Int[]
            chain_labels = String[]
            token_atom_names = Vector{Vector{String}}()
            token_atom_coords = Vector{Dict{String,NTuple{3,Float32}}}()
            design_mask = Bool[]
            binding_type = Int[]
            ss_type = Int[]
            structure_group = Int[]
            target_msa_mask = Bool[]
            cyclic_period = Int[]

            chain_aliases = Dict{String, Vector{String}}()
            used_chain_labels = Set{String}()

            function register_alias!(orig::String, effective::String)
                if !haskey(chain_aliases, orig)
                    chain_aliases[orig] = String[]
                end
                if !(effective in chain_aliases[orig])
                    push!(chain_aliases[orig], effective)
                end
            end

            function make_unique_chain_label(label::String)
                if !(label in used_chain_labels)
                    push!(used_chain_labels, label)
                    return label
                end
                i = 2
                while true
                    cand = "$(label)_$(i)"
                    if !(cand in used_chain_labels)
                        push!(used_chain_labels, cand)
                        return cand
                    end
                    i += 1
                end
            end

            function append_entity!(
                ent_residue_tokens,
                ent_mol_types,
                ent_residue_indices,
                ent_chain_labels,
                ent_token_atom_names,
                ent_token_atom_coords,
                ent_design_mask,
                ent_binding_type,
                ent_ss_type,
                ent_structure_group,
                ent_target_msa_mask,
                ent_cyclic_period,
                ent_sym_ids,
                orig_chain_names::Vector{String},
                fuse_target::Union{Nothing,String}=nothing,
            )
                local_chain_map = Dict{String, String}()
                if fuse_target === nothing
                    for cname in orig_chain_names
                        local_chain_map[cname] = make_unique_chain_label(cname)
                        register_alias!(cname, local_chain_map[cname])
                    end
                else
                    fused_chain = _resolve_constraint_chain(string(fuse_target), chain_aliases)
                    any(==(fused_chain), chain_labels) || error("Fuse target chain not found: $(fuse_target)")
                    for cname in orig_chain_names
                        local_chain_map[cname] = fused_chain
                        register_alias!(cname, fused_chain)
                    end
                end

                local_asym_map = Dict{String, Int}()
                if fuse_target === nothing
                    for cname in values(local_chain_map)
                        local_asym_map[cname] = length(local_asym_map)
                    end
                end

                next_asym_base = if fuse_target === nothing
                    isempty(asym_ids) ? 0 : (maximum(asym_ids) + 1)
                else
                    0
                end
                ent_id_base = if fuse_target === nothing
                    next_asym_base
                else
                    0
                end
                fused_chain_offsets = Dict{String, Int}()
                fused_chain_meta = Dict{String, NTuple{3,Int}}()  # asym, entity, sym

                for i in eachindex(ent_residue_tokens)
                    c_orig = ent_chain_labels[i]
                    c_eff = local_chain_map[c_orig]
                    a_global = 0
                    ent_global = 0
                    sym_global = 0
                    res_idx_out = ent_residue_indices[i]
                    if fuse_target === nothing
                        a_local = local_asym_map[c_eff]
                        a_global = next_asym_base + a_local
                        ent_global = ent_id_base + a_local
                        sym_global = ent_sym_ids[i]
                    else
                        if !haskey(fused_chain_meta, c_eff)
                            idxs = findall(==(c_eff), chain_labels)
                            !isempty(idxs) || error("Fuse target chain not found: $(c_eff)")
                            first_idx = first(idxs)
                            fused_chain_meta[c_eff] = (asym_ids[first_idx], entity_ids[first_idx], sym_ids[first_idx])
                            fused_chain_offsets[c_eff] = maximum(residue_indices[idxs]) + 1
                        end
                        a_global, ent_global, sym_global = fused_chain_meta[c_eff]
                        res_idx_out = fused_chain_offsets[c_eff]
                        fused_chain_offsets[c_eff] += 1
                    end

                    push!(residue_tokens, ent_residue_tokens[i])
                    push!(mol_types, ent_mol_types[i])
                    push!(asym_ids, a_global)
                    push!(entity_ids, ent_global)
                    push!(sym_ids, sym_global)
                    push!(residue_indices, res_idx_out)
                    push!(chain_labels, c_eff)
                    push!(token_atom_names, ent_token_atom_names[i])
                    push!(token_atom_coords, ent_token_atom_coords[i])
                    push!(design_mask, ent_design_mask[i])
                    push!(binding_type, ent_binding_type[i])
                    push!(ss_type, ent_ss_type[i])
                    push!(structure_group, ent_structure_group[i])
                    push!(target_msa_mask, ent_target_msa_mask[i])
                    push!(cyclic_period, ent_cyclic_period[i])
                end
            end

            entities = _as_list(_ydict_get(schema, "entities", Any[]))
            for e in entities
                if _ydict_get(e, "protein", nothing) !== nothing || _ydict_get(e, "dna", nothing) !== nothing || _ydict_get(e, "rna", nothing) !== nothing
                    chain_type = _ydict_get(e, "protein", nothing) !== nothing ? "PROTEIN" : (_ydict_get(e, "dna", nothing) !== nothing ? "DNA" : "RNA")
                    spec = _ydict_get(e, lowercase(chain_type), nothing)
                    ids = _as_list(_ydict_get(spec, "id", nothing))
                    isempty(ids) && error("Missing id for $chain_type entity")
                    seq = _ydict_get(spec, "sequence", nothing)
                    seq === nothing && error("Missing sequence for $chain_type entity")
                    toks, dmask = _parse_polymer_sequence(string(seq), chain_type, rng)
                    n = length(toks)
                    sym_group = Int(_ydict_get(spec, "symmetric_group", 0))
                    msa_flag = _as_boolish(_ydict_get(spec, "msa", false), false)
                    cyclic_flag = _as_boolish(_ydict_get(spec, "cyclic", false), false)
                    cyclic_val = cyclic_flag ? n : 0
                    fuse_target = _ydict_get(spec, "fuse", nothing)
                    btype = _parse_binding_spec(_ydict_get(spec, "binding_types", nothing), n)
                    sstype = _parse_ss_spec(_ydict_get(spec, "secondary_structure", nothing), n)

                    ent_residue_tokens = String[]
                    ent_mol_types = Int[]
                    ent_residue_indices = Int[]
                    ent_chain_labels = String[]
                    ent_token_atom_names = Vector{Vector{String}}()
                    ent_token_atom_coords = Vector{Dict{String,NTuple{3,Float32}}}()
                    ent_design_mask = Bool[]
                    ent_binding_type = Int[]
                    ent_ss_type = Int[]
                    ent_structure_group = Int[]
                    ent_target_msa_mask = Bool[]
                    ent_cyclic_period = Int[]
                    ent_sym_ids = Int[]

                    mt = chain_type_ids[chain_type]
                    for id_any in ids
                        cid = string(id_any)
                        append!(ent_residue_tokens, toks)
                        append!(ent_mol_types, fill(mt, n))
                        append!(ent_residue_indices, collect(1:n))
                        append!(ent_chain_labels, fill(cid, n))
                        append!(ent_token_atom_names, [String[] for _ in 1:n])
                        append!(ent_token_atom_coords, [Dict{String,NTuple{3,Float32}}() for _ in 1:n])
                        append!(ent_design_mask, dmask)
                        append!(ent_binding_type, btype)
                        append!(ent_ss_type, sstype)
                        append!(ent_structure_group, fill(0, n))
                        append!(ent_target_msa_mask, fill(msa_flag, n))
                        append!(ent_cyclic_period, fill(cyclic_val, n))
                        append!(ent_sym_ids, fill(sym_group, n))
                    end

                    append_entity!(
                        ent_residue_tokens,
                        ent_mol_types,
                        ent_residue_indices,
                        ent_chain_labels,
                        ent_token_atom_names,
                        ent_token_atom_coords,
                        ent_design_mask,
                        ent_binding_type,
                        ent_ss_type,
                        ent_structure_group,
                        ent_target_msa_mask,
                        ent_cyclic_period,
                        ent_sym_ids,
                        [string(x) for x in ids],
                        fuse_target === nothing ? nothing : string(fuse_target),
                    )

                elseif _ydict_get(e, "ligand", nothing) !== nothing
                    spec = _ydict_get(e, "ligand", nothing)
                    ids = _as_list(_ydict_get(spec, "id", nothing))
                    isempty(ids) && error("Missing id for ligand entity")
                    smiles = _ydict_get(spec, "smiles", nothing)
                    ccd = _ydict_get(spec, "ccd", nothing)
                    sym_group = Int(_ydict_get(spec, "symmetric_group", 0))
                    fuse_target = _ydict_get(spec, "fuse", nothing)

                    toks = String[]
                    atom_names_src = Vector{Vector{String}}()
                    atom_coords_src = Vector{Dict{String,NTuple{3,Float32}}}()
                    if smiles !== nothing
                        smiles_list = [String(strip(string(x))) for x in _as_list(smiles)]
                        lig = smiles_to_ligand_tokens(smiles_list)
                        toks = lig.tokens
                        atom_names_src = lig.token_atom_names
                        atom_coords_src = lig.token_atom_coords
                    elseif ccd !== nothing
                        ccds = _as_list(ccd)
                        toks = [haskey(token_ids, _upper(x)) ? _upper(x) : "UNK" for x in ccds]
                        atom_names_src = [String[] for _ in 1:length(toks)]
                        atom_coords_src = [Dict{String,NTuple{3,Float32}}() for _ in 1:length(toks)]
                    else
                        error("Ligand entity requires either 'smiles' or 'ccd'")
                    end

                    n = length(toks)
                    btype = _parse_binding_spec(_ydict_get(spec, "binding_types", nothing), n)
                    sstype = _parse_ss_spec(_ydict_get(spec, "secondary_structure", nothing), n)

                    ent_residue_tokens = String[]
                    ent_mol_types = Int[]
                    ent_residue_indices = Int[]
                    ent_chain_labels = String[]
                    ent_token_atom_names = Vector{Vector{String}}()
                    ent_token_atom_coords = Vector{Dict{String,NTuple{3,Float32}}}()
                    ent_design_mask = Bool[]
                    ent_binding_type = Int[]
                    ent_ss_type = Int[]
                    ent_structure_group = Int[]
                    ent_target_msa_mask = Bool[]
                    ent_cyclic_period = Int[]
                    ent_sym_ids = Int[]

                    mt = chain_type_ids["NONPOLYMER"]
                    for id_any in ids
                        cid = string(id_any)
                        append!(ent_residue_tokens, toks)
                        append!(ent_mol_types, fill(mt, n))
                        append!(ent_residue_indices, collect(1:n))
                        append!(ent_chain_labels, fill(cid, n))
                        append!(ent_token_atom_names, [copy(atom_names_src[i]) for i in 1:n])
                        append!(ent_token_atom_coords, [copy(atom_coords_src[i]) for i in 1:n])
                        append!(ent_design_mask, fill(false, n))
                        append!(ent_binding_type, btype)
                        append!(ent_ss_type, sstype)
                        append!(ent_structure_group, fill(0, n))
                        append!(ent_target_msa_mask, fill(false, n))
                        append!(ent_cyclic_period, fill(0, n))
                        append!(ent_sym_ids, fill(sym_group, n))
                    end

                    append_entity!(
                        ent_residue_tokens,
                        ent_mol_types,
                        ent_residue_indices,
                        ent_chain_labels,
                        ent_token_atom_names,
                        ent_token_atom_coords,
                        ent_design_mask,
                        ent_binding_type,
                        ent_ss_type,
                        ent_structure_group,
                        ent_target_msa_mask,
                        ent_cyclic_period,
                        ent_sym_ids,
                        [string(x) for x in ids],
                        fuse_target === nothing ? nothing : string(fuse_target),
                    )

                elseif _ydict_get(e, "file", nothing) !== nothing
                    spec = _ydict_get(e, "file", nothing)
                    parsed_file = _parse_file_entity(spec, base_dir, include_nonpolymer, rng)
                    orig_chain_names = unique(parsed_file.chain_labels)
                    fuse_target = _ydict_get(spec, "fuse", nothing)

                    append_entity!(
                        parsed_file.residue_tokens,
                        parsed_file.mol_types,
                        parsed_file.residue_indices,
                        parsed_file.chain_labels,
                        parsed_file.token_atom_names,
                        parsed_file.token_atom_coords,
                        parsed_file.design_mask,
                        parsed_file.binding_type,
                        parsed_file.ss_type,
                        parsed_file.structure_group,
                        parsed_file.target_msa_mask,
                        parsed_file.cyclic_period,
                        parsed_file.sym_ids,
                        orig_chain_names,
                        fuse_target === nothing ? nothing : string(fuse_target),
                    )
                else
                    error("Unsupported entity in YAML: $(collect(keys(e)))")
                end
            end

            T = length(residue_tokens)
            if total_len !== nothing
                if !(total_len.min <= T <= total_len.max)
                    continue
                end
            end

            token_idxs_by_chain = Dict{String, Vector{Int}}()
            for i in 1:T
                c = chain_labels[i]
                if !haskey(token_idxs_by_chain, c)
                    token_idxs_by_chain[c] = Int[]
                end
                push!(token_idxs_by_chain[c], i)
            end

            bonds = NTuple{3, Int}[]
            constraints = _ydict_get(schema, "constraints", nothing)
            if constraints !== nothing
                for c in _as_list(constraints)
                    b = _ydict_get(c, "bond", nothing)
                    b === nothing && continue
                    a1 = _as_list(_ydict_get(b, "atom1", nothing))
                    a2 = _as_list(_ydict_get(b, "atom2", nothing))
                    length(a1) >= 2 || continue
                    length(a2) >= 2 || continue

                    c1 = _resolve_constraint_chain(string(a1[1]), chain_aliases)
                    c2 = _resolve_constraint_chain(string(a2[1]), chain_aliases)
                    r1 = Int(a1[2])
                    r2 = Int(a2[2])

                    haskey(token_idxs_by_chain, c1) || error("Constraint chain not found: $c1")
                    haskey(token_idxs_by_chain, c2) || error("Constraint chain not found: $c2")
                    idxs1 = token_idxs_by_chain[c1]
                    idxs2 = token_idxs_by_chain[c2]
                    1 <= r1 <= length(idxs1) || error("Constraint residue index out of range for chain $c1: $r1")
                    1 <= r2 <= length(idxs2) || error("Constraint residue index out of range for chain $c2: $r2")

                    t1 = idxs1[r1]
                    t2 = idxs2[r2]
                    bt_name = _upper(_ydict_get(b, "bondtype", "COVALENT"))
                    bt_id = get(bond_type_ids, bt_name, get(bond_type_ids, "COVALENT", 1))
                    push!(bonds, (t1, t2, bt_id))
                end
            end

            leaving_atoms = _ydict_get(schema, "leaving_atoms", nothing)
            if leaving_atoms !== nothing
                for e in _as_list(leaving_atoms)
                    a = _as_list(_ydict_get(e, "atom", nothing))
                    length(a) >= 3 || continue
                    c = _resolve_constraint_chain(string(a[1]), chain_aliases)
                    ridx = Int(a[2])
                    atom_name = strip(string(a[3]))

                    haskey(token_idxs_by_chain, c) || error("Leaving-atom chain not found: $c")
                    idxs = token_idxs_by_chain[c]
                    1 <= ridx <= length(idxs) || error("Leaving-atom residue index out of range for chain $c: $ridx")
                    t = idxs[ridx]

                    # Remove both exact and upper-cased aliases for robustness.
                    names = token_atom_names[t]
                    upper_atom_name = uppercase(atom_name)
                    deleteat!(names, findall(n -> uppercase(strip(n)) == upper_atom_name, names))

                    coords = token_atom_coords[t]
                    for key in collect(keys(coords))
                        if uppercase(strip(key)) == upper_atom_name
                            delete!(coords, key)
                        end
                    end
                end
            end

            return (
                residue_tokens=residue_tokens,
                mol_types=mol_types,
                asym_ids=asym_ids,
                entity_ids=entity_ids,
                sym_ids=sym_ids,
                residue_indices=residue_indices,
                chain_labels=chain_labels,
                token_atom_names=token_atom_names,
                token_atom_coords=token_atom_coords,
                design_mask=design_mask,
                binding_type=binding_type,
                ss_type=ss_type,
                structure_group=structure_group,
                target_msa_mask=target_msa_mask,
                cyclic_period=cyclic_period,
                bonds=bonds,
            )
        catch err
            if total_len === nothing
                rethrow(err)
            end
            last_err = err
        end
    end

    if last_err !== nothing
        throw(last_err)
    end
    error("Failed to parse YAML after $(max_total_len_trials) attempts")
end
