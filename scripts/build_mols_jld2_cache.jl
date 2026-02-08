using Dates
using JSON3
using JLD2

const BOND_TYPE_IDS = Dict(
    "OTHER" => 1,
    "SINGLE" => 2,
    "DOUBLE" => 3,
    "TRIPLE" => 4,
    "AROMATIC" => 5,
    "COVALENT" => 6,
    "DATIVE" => 1,
)

function parse_kv_args(args)
    out = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if startswith(key, "--")
            key = key[3:end]
            if i < length(args) && !startswith(args[i + 1], "--")
                out[key] = args[i + 1]
                i += 2
            else
                out[key] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return out
end

function _as_string_vec(x, label::AbstractString)
    x isa AbstractVector || error("$label must be a JSON array")
    out = String[]
    for v in x
        s = strip(string(v))
        isempty(s) && error("$label contains empty string")
        push!(out, s)
    end
    return out
end

function _as_int_vec(x, label::AbstractString)
    x isa AbstractVector || error("$label must be a JSON array")
    out = Int[]
    for v in x
        push!(out, Int(v))
    end
    return out
end

function _as_coords(x, n_atoms::Int)
    x isa AbstractVector || error("coords must be a JSON array")
    length(x) == n_atoms || error("coords length mismatch: got $(length(x)), expected $n_atoms")
    coords = zeros(Float32, 3, n_atoms)
    for i in 1:n_atoms
        row = x[i]
        row isa AbstractVector || error("coords[$i] must be a 3-vector")
        length(row) == 3 || error("coords[$i] length mismatch: got $(length(row)), expected 3")
        coords[1, i] = Float32(row[1])
        coords[2, i] = Float32(row[2])
        coords[3, i] = Float32(row[3])
    end
    return coords
end

function _as_bonds(x, n_atoms::Int)
    x isa AbstractVector || error("bonds must be a JSON array")
    bond_i = Int32[]
    bond_j = Int32[]
    bond_type = Int16[]
    seen = Dict{Tuple{Int,Int}, Int16}()
    for (k, item) in enumerate(x)
        item isa AbstractVector || error("bonds[$k] must be a 3-array [i,j,type]")
        length(item) == 3 || error("bonds[$k] length mismatch: got $(length(item)), expected 3")
        i = Int(item[1])
        j = Int(item[2])
        1 <= i <= n_atoms || error("bonds[$k] i out of range: $i (n_atoms=$n_atoms)")
        1 <= j <= n_atoms || error("bonds[$k] j out of range: $j (n_atoms=$n_atoms)")
        i == j && error("bonds[$k] is a self-bond at atom $i")
        a, b = i < j ? (i, j) : (j, i)
        bt_name = uppercase(strip(string(item[3])))
        haskey(BOND_TYPE_IDS, bt_name) || error("bonds[$k] unknown bond type: $bt_name")
        bt = Int16(BOND_TYPE_IDS[bt_name])
        if haskey(seen, (a, b))
            seen[(a, b)] == bt || error("inconsistent bond type for edge ($a,$b)")
            continue
        end
        seen[(a, b)] = bt
        push!(bond_i, Int32(a))
        push!(bond_j, Int32(b))
        push!(bond_type, bt)
    end
    return bond_i, bond_j, bond_type
end

function build_cache(jsonl_path::AbstractString, out_path::AbstractString)
    isfile(jsonl_path) || error("JSONL input not found: $jsonl_path")
    mkpath(dirname(out_path))

    molecule_ids = String[]
    seen_ids = Set{String}()
    n_records = 0

    jldopen(out_path, "w") do f
        f["meta/format_version"] = Int32(1)
        f["meta/created_utc"] = string(Dates.now(Dates.UTC))
        f["meta/source"] = "boltzgen moldir pkl export"

        open(jsonl_path, "r") do io
            for (lineno, raw_line) in enumerate(eachline(io))
                line = strip(raw_line)
                isempty(line) && error("Encountered empty JSONL line at $lineno")
                record = JSON3.read(line)

                code = uppercase(strip(string(record.id)))
                isempty(code) && error("Line $lineno has empty molecule id")
                code in seen_ids && error("Duplicate molecule id '$code' at line $lineno")
                push!(seen_ids, code)
                push!(molecule_ids, code)

                atom_names = _as_string_vec(record.atom_names, "atom_names")
                elements = _as_int_vec(record.elements, "elements")
                charges = _as_int_vec(record.charges, "charges")
                n_atoms = length(atom_names)
                length(elements) == n_atoms || error("$code elements length mismatch")
                length(charges) == n_atoms || error("$code charges length mismatch")
                coords = _as_coords(record.coords, n_atoms)
                bond_i, bond_j, bond_type = _as_bonds(record.bonds, n_atoms)

                grp = "molecules/$code"
                f["$grp/atom_names"] = atom_names
                f["$grp/elements"] = Int16.(elements)
                f["$grp/charges"] = Int16.(charges)
                f["$grp/ref_coords"] = coords
                f["$grp/bond_i"] = bond_i
                f["$grp/bond_j"] = bond_j
                f["$grp/bond_type"] = bond_type

                n_records += 1
                if n_records % 1000 == 0
                    println("processed $n_records records")
                end
            end
        end

        n_records > 0 || error("No records were written to cache")
        f["meta/count"] = Int32(n_records)
        f["molecule_ids"] = molecule_ids
    end

    println("Wrote JLD2 cache: $out_path (records=$n_records)")
end

function main()
    args = parse_kv_args(ARGS)
    in_jsonl = get(args, "jsonl", "")
    out_path = get(args, "out", "")
    isempty(in_jsonl) && error("Missing --jsonl <path>")
    isempty(out_path) && error("Missing --out <path>")
    build_cache(abspath(in_jsonl), abspath(out_path))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
