#!/usr/bin/env julia

function parse_pdb_coords(path::AbstractString)
    coords = NTuple{3,Float32}[]
    open(path, "r") do io
        for ln in eachline(io)
            if startswith(ln, "ATOM") || startswith(ln, "HETATM")
                if ncodeunits(ln) < 54
                    continue
                end
                x = tryparse(Float32, strip(SubString(ln, 31, 38)))
                y = tryparse(Float32, strip(SubString(ln, 39, 46)))
                z = tryparse(Float32, strip(SubString(ln, 47, 54)))
                if x === nothing || y === nothing || z === nothing
                    continue
                end
                if isfinite(x) && isfinite(y) && isfinite(z)
                    push!(coords, (x, y, z))
                end
            end
        end
    end
    return coords
end

function geometry_stats(coords::Vector{NTuple{3,Float32}})
    n = length(coords)
    if n == 0
        return (
            n_atoms=0,
            max_abs=0f0,
            min_nn=Inf32,
            max_nn=Inf32,
            frac_abs_ge900=0f0,
        )
    end

    max_abs = 0f0
    far = 0
    nn_d2 = fill(Inf32, n)
    for i in 1:n
        x, y, z = coords[i]
        local_abs = max(abs(x), max(abs(y), abs(z)))
        max_abs = max(max_abs, local_abs)
        if local_abs >= 900f0
            far += 1
        end
    end

    if n >= 2
        for i in 1:(n - 1)
            xi, yi, zi = coords[i]
            for j in (i + 1):n
                xj, yj, zj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                d2 = dx * dx + dy * dy + dz * dz
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
    return (
        n_atoms=n,
        max_abs=max_abs,
        min_nn=minimum(nn),
        max_nn=maximum(nn),
        frac_abs_ge900=Float32(far) / Float32(n),
    )
end

function check_geometry(path::AbstractString)
    coords = parse_pdb_coords(path)
    st = geometry_stats(coords)
    failures = String[]
    st.n_atoms < 4 && push!(failures, "too few atoms ($(st.n_atoms))")
    st.max_abs > 200f0 && push!(failures, "max_abs=$(st.max_abs) > 200.0")
    st.frac_abs_ge900 > 0.05f0 && push!(failures, "frac_abs_ge900=$(st.frac_abs_ge900) > 0.05")
    isfinite(st.min_nn) && st.min_nn < 0.01f0 && push!(failures, "min_nn=$(st.min_nn) < 0.01")
    isfinite(st.max_nn) && st.max_nn > 8.0f0 && push!(failures, "max_nn=$(st.max_nn) > 8.0")
    return st, failures
end

function main(paths::Vector{String})
    isempty(paths) && error("Usage: julia check_output_pdb_geometry.jl <pdb-path> [<pdb-path> ...]")
    any_failed = false
    for p in paths
        st, failures = check_geometry(p)
        if isempty(failures)
            println("PASS\t", p, "\t", st)
        else
            any_failed = true
            println("FAIL\t", p, "\t", join(failures, "; "), "\t", st)
        end
    end
    any_failed && error("Geometry checks failed for one or more PDB files.")
end

main(String.(ARGS))
