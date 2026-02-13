using Printf
using LinearAlgebra

const atom14_placement_count_to_token = Dict(
    (0, 0, 0, 9) => "ALA",
    (3, 0, 0, 0) => "ARG",
    (1, 0, 0, 5) => "ASN",
    (2, 0, 0, 4) => "ASP",
    (0, 0, 0, 8) => "CYS",
    (0, 0, 0, 5) => "GLN",
    (2, 0, 0, 3) => "GLU",
    (0, 0, 0, 10) => "GLY",
    (0, 0, 0, 4) => "HIS",
    (0, 0, 0, 6) => "ILE",
    (4, 0, 0, 2) => "LEU",
    (5, 0, 0, 0) => "LYS",
    (6, 0, 0, 0) => "MET",
    (0, 0, 0, 3) => "PHE",
    (0, 0, 0, 7) => "PRO",
    (8, 0, 0, 0) => "SER",
    (3, 0, 0, 4) => "THR",
    (0, 0, 0, 0) => "TRP",
    (0, 0, 0, 2) => "TYR",
    (7, 0, 0, 0) => "VAL",
    (9, 0, 0, 0) => "UNK",
)

const atom14_ref_atoms = Dict(
    "ALA" => ["N", "CA", "C", "O", "CB"],
    "ARG" => ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS" => ["N", "CA", "C", "O", "CB", "SG"],
    "GLN" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY" => ["N", "CA", "C", "O"],
    "HIS" => ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS" => ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET" => ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO" => ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER" => ["N", "CA", "C", "O", "CB", "OG"],
    "THR" => ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL" => ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "UNK" => ["N", "CA", "C", "O", "CB"],
)

const simple_element_to_atomic_num = Dict(
    "H" => 1,
    "C" => 6,
    "N" => 7,
    "O" => 8,
    "P" => 15,
    "S" => 16,
)

const PDB_COORD_MIN = -999.999f0
const PDB_COORD_MAX = 9999.999f0

_clip_pdb_coord(x::Real) = clamp(Float32(x), PDB_COORD_MIN, PDB_COORD_MAX)

function _res_name_from_onehot(res_type_vec)
    idx = argmax(res_type_vec)
    return tokens[idx]
end

function _element_from_atom_name(atom_name::String)
    stripped = replace(atom_name, r"[0-9]" => "")
    stripped = strip(stripped)
    isempty(stripped) && return "X"
    return uppercase(stripped[1:1])
end

function _is_fake_or_mask_atom(atom_name::String)
    u = uppercase(strip(atom_name))
    # Match upstream mmCIF writer behavior: skip fake/masked atom labels.
    return startswith(u, "LV") || startswith(u, "FL")
end

function _elem_from_name_simple(atom_name::String)
    stripped = replace(atom_name, r"[0-9]" => "")
    stripped = strip(stripped)
    isempty(stripped) && return "C"
    return uppercase(stripped[1:1])
end

function postprocess_atom14(feats::Dict, coords; threshold::Float32=0.5f0, invalid_token::String="UNK")
    # This mirrors python's res_from_atom14 post-processing for design outputs.
    new = copy(feats)
    res_type = copy(feats["res_type"])
    ref_atom_name_chars = copy(feats["ref_atom_name_chars"])
    ref_element = copy(feats["ref_element"])

    token_n = size(res_type, 2)
    batch_n = size(res_type, 3)
    atom_n = size(feats["atom_pad_mask"], 1)
    protein_id = chain_type_ids["PROTEIN"]

    coords_batched = ndims(coords) == 2 ? reshape(coords, size(coords, 1), size(coords, 2), 1) : coords

    for b in 1:batch_n
        token_design = (feats["design_mask"][:, b] .> 0.5) .& (feats["mol_type"][:, b] .== protein_id)
        any(token_design) || continue

        # Map each atom to its token by one-hot argmax.
        atom_to_token = feats["atom_to_token"][:, :, b]
        atom_token_idx = [argmax(view(atom_to_token, m, :)) for m in 1:atom_n]
        atom_pad = feats["atom_pad_mask"][:, b] .> 0.5

        for t in 1:token_n
            token_design[t] || continue
            atom_idxs = [m for m in 1:atom_n if atom_pad[m] && atom_token_idx[m] == t]
            length(atom_idxs) < 14 && continue
            atom_idxs = atom_idxs[1:14]

            c = permutedims(coords_batched[:, atom_idxs, b], (2, 1)) # (14, 3)
            bb = view(c, 1:4, :)
            side = view(c, 5:14, :)
            counts = zeros(Int, 4)
            for j in 1:size(side, 1)
                d = [norm(view(side, j, :) .- view(bb, i, :)) for i in 1:4]
                min_d, min_i = findmin(d)
                min_d <= threshold || continue
                counts[min_i] += 1
            end

            token_name = get(atom14_placement_count_to_token, Tuple(counts), invalid_token)
            token_idx = get(token_ids, token_name, token_ids["UNK"])
            res_type[:, t, b] .= 0f0
            res_type[token_idx, t, b] = 1f0

            atom_names = get(atom14_ref_atoms, token_name, atom14_ref_atoms["UNK"])
            for (i, atom_name) in enumerate(atom_names)
                m = atom_idxs[i]
                ref_atom_name_chars[:, :, m, b] .= encode_atom_name_chars(atom_name)
                ref_element[:, m, b] .= 0f0
                elem = _elem_from_name_simple(atom_name)
                z = get(simple_element_to_atomic_num, elem, 6)
                zidx = z + 1
                if 1 <= zidx <= size(ref_element, 1)
                    ref_element[zidx, m, b] = 1f0
                end
            end
        end
    end

    new["res_type"] = res_type
    new["ref_atom_name_chars"] = ref_atom_name_chars
    new["ref_element"] = ref_element
    return new
end

function write_pdb(io::IO, feats::Dict, coords; batch::Int=1)
    # coords: (3, M) or (3, M, B)
    coords_b = ndims(coords) == 2 ? coords : coords[:, :, batch]

    atom_pad_mask = feats["atom_pad_mask"][:, batch]
    atom_to_token = feats["atom_to_token"][:, :, batch]
    ref_atom_name_chars = feats["ref_atom_name_chars"][:, :, :, batch]
    res_type, residue_index, asym_id, mol_type, token_pad_mask, res_offset = _token_metadata(feats, batch)

    atom_serial = 1
    M = size(coords_b, 2)
    skipped = 0
    clipped = 0
    for m in 1:M
        atom_pad_mask[m] > 0.5 || continue
        token_idx = argmax(view(atom_to_token, m, :))
        token_pad_mask[token_idx] > 0.5 || (skipped += 1; continue)
        res_name = _res_name_from_onehot(view(res_type, :, token_idx))
        res_seq = Int(residue_index[token_idx]) + res_offset
        chain_id = chain_id_from_asym(Int(asym_id[token_idx]))
        atom_name = decode_atom_name_chars(view(ref_atom_name_chars, :, :, m))
        _is_fake_or_mask_atom(atom_name) && (skipped += 1; continue)
        element = _element_from_atom_name(atom_name)
        record = (Int(mol_type[token_idx]) == chain_type_ids["NONPOLYMER"]) ? "HETATM" : "ATOM"

        x_raw = coords_b[1, m]
        y_raw = coords_b[2, m]
        z_raw = coords_b[3, m]
        x = _clip_pdb_coord(x_raw)
        y = _clip_pdb_coord(y_raw)
        z = _clip_pdb_coord(z_raw)
        if !isfinite(x) || !isfinite(y) || !isfinite(z)
            skipped += 1
            continue
        end
        if x != x_raw || y != y_raw || z != z_raw
            clipped += 1
        end

        @printf(
            io,
            "%-6s%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s\n",
            record,
            atom_serial,
            atom_name,
            res_name,
            chain_id,
            res_seq,
            x,
            y,
            z,
            1.00,
            0.00,
            element,
        )
        atom_serial += 1
    end
    if skipped > 0
        @printf(io, "REMARK Skipped %d atoms due to padding/NaNs\n", skipped)
    end
    if clipped > 0
        @printf(io, "REMARK Clipped %d coordinates to PDB range [%.3f, %.3f]\n", clipped, PDB_COORD_MIN, PDB_COORD_MAX)
    end
    println(io, "END")
end

function write_pdb(path::AbstractString, feats::Dict, coords; batch::Int=1)
    mkpath(dirname(path))
    open(io -> write_pdb(io, feats, coords; batch=batch), path, "w")
end

function _token_metadata(feats::Dict, batch::Int)
    res_type = feats["res_type"][:, :, batch]
    residue_index = if haskey(feats, "residue_index")
        feats["residue_index"][:, batch]
    elseif haskey(feats, "feature_residue_index")
        feats["feature_residue_index"][:, batch]
    else
        feats["token_index"][:, batch]
    end
    asym_id = feats["asym_id"][:, batch]
    mol_type = feats["mol_type"][:, batch]
    token_pad_mask = haskey(feats, "token_pad_mask") ? feats["token_pad_mask"][:, batch] : ones(Float32, size(res_type, 2))
    res_offset = minimum(residue_index) <= 0 ? 1 : 0
    return res_type, residue_index, asym_id, mol_type, token_pad_mask, res_offset
end

function collect_atom37_entries(feats::Dict, coords; batch::Int=1)
    coords_b = ndims(coords) == 2 ? coords : coords[:, :, batch]
    atom_pad_mask = feats["atom_pad_mask"][:, batch]
    atom_to_token = feats["atom_to_token"][:, :, batch]
    ref_atom_name_chars = feats["ref_atom_name_chars"][:, :, :, batch]
    res_type, residue_index, asym_id, mol_type, token_pad_mask, res_offset = _token_metadata(feats, batch)

    T = size(atom_to_token, 2)
    token_atom_map = [Dict{String,NTuple{3,Float32}}() for _ in 1:T]

    for m in 1:size(coords_b, 2)
        atom_pad_mask[m] > 0.5 || continue
        token_idx = argmax(view(atom_to_token, m, :))
        token_pad_mask[token_idx] > 0.5 || continue

        atom_name = decode_atom_name_chars(view(ref_atom_name_chars, :, :, m))
        _is_fake_or_mask_atom(atom_name) && continue

        x = coords_b[1, m]
        y = coords_b[2, m]
        z = coords_b[3, m]
        (isfinite(x) && isfinite(y) && isfinite(z)) || continue

        token_atom_map[token_idx][atom_name] = (Float32(x), Float32(y), Float32(z))
    end

    entries = NamedTuple[]
    for t in 1:T
        token_pad_mask[t] > 0.5 || continue
        res_name = _res_name_from_onehot(view(res_type, :, t))
        res_seq = Int(residue_index[t]) + res_offset
        chain_id = chain_id_from_asym(Int(asym_id[t]))
        record = (Int(mol_type[t]) == chain_type_ids["NONPOLYMER"]) ? "HETATM" : "ATOM"
        amap = token_atom_map[t]

        ordered_atom_names = if Int(mol_type[t]) == chain_type_ids["PROTEIN"]
            atom_types
        else
            get(ref_atoms, res_name, collect(keys(amap)))
        end

        for atom_name in ordered_atom_names
            if haskey(amap, atom_name)
                xyz = amap[atom_name]
                push!(entries, (
                    record=record,
                    atom_name=atom_name,
                    res_name=res_name,
                    chain_id=chain_id,
                    res_seq=res_seq,
                    x=xyz[1],
                    y=xyz[2],
                    z=xyz[3],
                    element=_element_from_atom_name(atom_name),
                ))
            end
        end
    end

    return entries
end

function write_pdb_atom37(io::IO, feats::Dict, coords; batch::Int=1)
    entries = collect_atom37_entries(feats, coords; batch=batch)
    clipped = 0
    for (i, e) in enumerate(entries)
        x = _clip_pdb_coord(e.x)
        y = _clip_pdb_coord(e.y)
        z = _clip_pdb_coord(e.z)
        if x != e.x || y != e.y || z != e.z
            clipped += 1
        end
        @printf(
            io,
            "%-6s%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s\n",
            e.record,
            i,
            e.atom_name,
            e.res_name,
            e.chain_id,
            e.res_seq,
            x,
            y,
            z,
            1.00,
            0.00,
            e.element,
        )
    end
    if clipped > 0
        @printf(io, "REMARK Clipped %d coordinates to PDB range [%.3f, %.3f]\n", clipped, PDB_COORD_MIN, PDB_COORD_MAX)
    end
    println(io, "END")
end

function write_pdb_atom37(path::AbstractString, feats::Dict, coords; batch::Int=1)
    mkpath(dirname(path))
    open(io -> write_pdb_atom37(io, feats, coords; batch=batch), path, "w")
end

function write_mmcif(io::IO, feats::Dict, coords; batch::Int=1)
    entries = collect_atom37_entries(feats, coords; batch=batch)
    println(io, "data_generated")
    println(io, "#")
    println(io, "loop_")
    println(io, "_atom_site.group_PDB")
    println(io, "_atom_site.id")
    println(io, "_atom_site.type_symbol")
    println(io, "_atom_site.label_atom_id")
    println(io, "_atom_site.label_comp_id")
    println(io, "_atom_site.label_asym_id")
    println(io, "_atom_site.label_seq_id")
    println(io, "_atom_site.Cartn_x")
    println(io, "_atom_site.Cartn_y")
    println(io, "_atom_site.Cartn_z")
    println(io, "_atom_site.occupancy")
    println(io, "_atom_site.B_iso_or_equiv")
    println(io, "_atom_site.pdbx_PDB_model_num")
    for (i, e) in enumerate(entries)
        @printf(
            io,
            "%s %d %s %s %s %s %d %.3f %.3f %.3f %.2f %.2f %d\n",
            e.record,
            i,
            e.element,
            e.atom_name,
            e.res_name,
            string(e.chain_id),
            e.res_seq,
            e.x,
            e.y,
            e.z,
            1.00,
            0.00,
            1,
        )
    end
    println(io, "#")
end

function write_mmcif(path::AbstractString, feats::Dict, coords; batch::Int=1)
    mkpath(dirname(path))
    open(io -> write_mmcif(io, feats, coords; batch=batch), path, "w")
end

function geometry_stats_atom37(feats::Dict, coords; batch::Int=1)
    entries = collect_atom37_entries(feats, coords; batch=batch)
    n = length(entries)
    if n == 0
        return (
            n_atoms=0,
            max_abs=0f0,
            min_nearest_neighbor=Inf32,
            max_nearest_neighbor=Inf32,
            frac_nearest_lt_0p5=0f0,
            frac_abs_ge900=0f0,
        )
    end

    xyz = Matrix{Float32}(undef, 3, n)
    far = 0
    max_abs = 0f0
    for i in 1:n
        x = Float32(entries[i].x)
        y = Float32(entries[i].y)
        z = Float32(entries[i].z)
        xyz[1, i] = x
        xyz[2, i] = y
        xyz[3, i] = z

        local_abs = max(abs(x), max(abs(y), abs(z)))
        max_abs = max(max_abs, local_abs)
        if local_abs >= 900f0
            far += 1
        end
    end

    min_d2 = Inf32
    nn_d2 = fill(Inf32, n)
    if n >= 2
        for i in 1:(n - 1)
            xi = xyz[1, i]
            yi = xyz[2, i]
            zi = xyz[3, i]
            for j in (i + 1):n
                dx = xi - xyz[1, j]
                dy = yi - xyz[2, j]
                dz = zi - xyz[3, j]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < min_d2
                    min_d2 = d2
                end
                if d2 < nn_d2[i]
                    nn_d2[i] = d2
                end
                if d2 < nn_d2[j]
                    nn_d2[j] = d2
                end
            end
        end
    end

    nn = sqrt.(nn_d2)
    max_nn = isfinite(minimum(nn)) ? maximum(nn) : Inf32
    frac_nn_lt_0p5 = Float32(count(x -> isfinite(x) && x < 0.5f0, nn)) / Float32(n)

    return (
        n_atoms=n,
        max_abs=max_abs,
        min_nearest_neighbor=sqrt(min_d2),
        max_nearest_neighbor=max_nn,
        frac_nearest_lt_0p5=frac_nn_lt_0p5,
        frac_abs_ge900=Float32(far) / Float32(n),
    )
end

function assert_geometry_sane_atom37!(
    feats::Dict,
    coords;
    batch::Int=1,
    max_abs::Float32=200f0,
    max_far_coord_fraction::Float32=0.05f0,
    min_min_nearest_neighbor::Float32=0.01f0,
    max_max_nearest_neighbor::Float32=8.0f0,
    max_frac_nearest_lt_0p5::Union{Nothing,Float32}=nothing,
)
    stats = geometry_stats_atom37(feats, coords; batch=batch)
    failures = String[]
    if stats.n_atoms < 4
        push!(failures, "too few atoms ($(stats.n_atoms))")
    end
    if stats.max_abs > max_abs
        push!(failures, "max_abs=$(stats.max_abs) > $max_abs")
    end
    if stats.frac_abs_ge900 > max_far_coord_fraction
        push!(failures, "frac_abs_ge900=$(stats.frac_abs_ge900) > $max_far_coord_fraction")
    end
    if isfinite(stats.min_nearest_neighbor) && stats.min_nearest_neighbor < min_min_nearest_neighbor
        push!(failures, "min_nearest_neighbor=$(stats.min_nearest_neighbor) < $min_min_nearest_neighbor")
    end
    if isfinite(stats.max_nearest_neighbor) && stats.max_nearest_neighbor > max_max_nearest_neighbor
        push!(failures, "max_nearest_neighbor=$(stats.max_nearest_neighbor) > $max_max_nearest_neighbor")
    end
    if max_frac_nearest_lt_0p5 !== nothing && stats.frac_nearest_lt_0p5 > max_frac_nearest_lt_0p5
        push!(failures, "frac_nearest_lt_0p5=$(stats.frac_nearest_lt_0p5) > $max_frac_nearest_lt_0p5")
    end

    if !isempty(failures)
        error(
            "Atom37 geometry sanity check failed: " *
            join(failures, "; ") *
            ". stats=(n_atoms=$(stats.n_atoms), max_abs=$(stats.max_abs), min_nn=$(stats.min_nearest_neighbor), max_nn=$(stats.max_nearest_neighbor), frac_nn_lt_0p5=$(stats.frac_nearest_lt_0p5), frac_abs_ge900=$(stats.frac_abs_ge900))",
        )
    end

    return stats
end
