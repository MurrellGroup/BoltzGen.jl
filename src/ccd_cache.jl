using JLD2

const _ccd_cache_lock = ReentrantLock()
const _ccd_cache_open_path = Ref{Union{Nothing,String}}(nothing)
const _ccd_cache_open_file = Ref{Any}(nothing)
const _ccd_entry_cache = Dict{Tuple{String,String}, NamedTuple}()
const BOLTZGEN_MOLS_HF_REPO_ID = "MurrellLab/BoltzGen.jl"
const BOLTZGEN_MOLS_HF_REPO_TYPE = "datasets"
const BOLTZGEN_MOLS_HF_FILENAME = "mols_cache.jld2"

function _default_ccd_cache_path()
    # Local cache path overrides intentionally disabled.
    # Previous local-path logic:
    # if haskey(ENV, "BOLTZGEN_MOLS_JLD2")
    #     p = strip(String(ENV["BOLTZGEN_MOLS_JLD2"]))
    #     isempty(p) && error("ENV BOLTZGEN_MOLS_JLD2 is set but empty")
    #     return abspath(p)
    # end
    return hf_hub_download(
        BOLTZGEN_MOLS_HF_REPO_ID,
        BOLTZGEN_MOLS_HF_FILENAME;
        repo_type=BOLTZGEN_MOLS_HF_REPO_TYPE,
    )
end

function _resolve_ccd_cache_path(path::Union{Nothing,AbstractString})
    if path !== nothing
        error(
            "Local CCD cache paths are disabled. Set no cache_path and use the HuggingFace dataset " *
            "$(BOLTZGEN_MOLS_HF_REPO_ID) / $(BOLTZGEN_MOLS_HF_FILENAME).",
        )
    end
    if haskey(ENV, "BOLTZGEN_MOLS_JLD2")
        error(
            "ENV BOLTZGEN_MOLS_JLD2 is no longer supported. CCD cache is downloaded from HuggingFace dataset " *
            "$(BOLTZGEN_MOLS_HF_REPO_ID).",
        )
    end
    return _default_ccd_cache_path()
end

function _open_ccd_cache(cache_path::AbstractString)
    p = abspath(cache_path)
    isfile(p) || error(
        "CCD cache file not found after HuggingFace download: $p. " *
        "Expected dataset file $(BOLTZGEN_MOLS_HF_FILENAME) in $(BOLTZGEN_MOLS_HF_REPO_ID).",
    )

    lock(_ccd_cache_lock) do
        if _ccd_cache_open_file[] === nothing || _ccd_cache_open_path[] != p
            if _ccd_cache_open_file[] !== nothing
                close(_ccd_cache_open_file[])
            end
            f = jldopen(p, "r")
            haskey(f, "meta/format_version") || error("CCD cache missing meta/format_version: $p")
            haskey(f, "molecule_ids") || error("CCD cache missing molecule_ids: $p")
            _ccd_cache_open_file[] = f
            _ccd_cache_open_path[] = p
        end
        return _ccd_cache_open_file[]
    end
end

function load_ccd_ligand_cache_entry(code::AbstractString; cache_path::Union{Nothing,AbstractString}=nothing)
    c = uppercase(strip(String(code)))
    isempty(c) && error("CCD code is empty")
    p = _resolve_ccd_cache_path(cache_path)
    key = (p, c)
    if haskey(_ccd_entry_cache, key)
        return _ccd_entry_cache[key]
    end

    f = _open_ccd_cache(p)
    base = "molecules/$c"
    haskey(f, "$base/atom_names") || error("CCD '$c' not found in cache: $p")
    haskey(f, "$base/ref_coords") || error("CCD '$c' missing ref_coords in cache: $p")
    haskey(f, "$base/bond_i") || error("CCD '$c' missing bond_i in cache: $p")
    haskey(f, "$base/bond_j") || error("CCD '$c' missing bond_j in cache: $p")
    haskey(f, "$base/bond_type") || error("CCD '$c' missing bond_type in cache: $p")

    atom_names = Vector{String}(f["$base/atom_names"])
    n_atoms = length(atom_names)
    n_atoms > 0 || error("CCD '$c' has zero heavy atoms in cache: $p")
    length(Set(atom_names)) == n_atoms || error("CCD '$c' has duplicate atom names in cache: $p")

    ref_coords_raw = Array(f["$base/ref_coords"])
    ndims(ref_coords_raw) == 2 || error("CCD '$c' ref_coords must be 2D, got ndims=$(ndims(ref_coords_raw))")
    size(ref_coords_raw, 1) == 3 || error("CCD '$c' ref_coords first dimension must be 3, got $(size(ref_coords_raw, 1))")
    size(ref_coords_raw, 2) == n_atoms || error("CCD '$c' ref_coords atom dimension mismatch: $(size(ref_coords_raw, 2)) vs $n_atoms")
    ref_coords = Float32.(ref_coords_raw)

    atom_ref_coords = Dict{String,NTuple{3,Float32}}()
    for i in 1:n_atoms
        atom_ref_coords[atom_names[i]] = (
            ref_coords[1, i],
            ref_coords[2, i],
            ref_coords[3, i],
        )
    end

    bond_i = Int.(Array(f["$base/bond_i"]))
    bond_j = Int.(Array(f["$base/bond_j"]))
    bond_type = Int.(Array(f["$base/bond_type"]))
    length(bond_i) == length(bond_j) || error("CCD '$c' bond_i/bond_j length mismatch")
    length(bond_i) == length(bond_type) || error("CCD '$c' bond_i/bond_type length mismatch")

    bond_map = Dict{Tuple{Int,Int}, Int}()
    for k in eachindex(bond_i)
        i = bond_i[k]
        j = bond_j[k]
        bt = bond_type[k]
        1 <= i <= n_atoms || error("CCD '$c' bond_i out of range at index $k: $i")
        1 <= j <= n_atoms || error("CCD '$c' bond_j out of range at index $k: $j")
        i == j && error("CCD '$c' self-bond at index $k: atom $i")
        bt > 0 || error("CCD '$c' bond type must be > 0 at index $k, got $bt")
        a, b = i < j ? (i, j) : (j, i)
        if haskey(bond_map, (a, b))
            bond_map[(a, b)] == bt || error("CCD '$c' inconsistent bond type for edge ($a,$b)")
        else
            bond_map[(a, b)] = bt
        end
    end

    bonds = NTuple{3,Int}[]
    for (edge, bt) in sort(collect(bond_map); by=x -> (x[1][1], x[1][2]))
        push!(bonds, (edge[1], edge[2], bt))
    end

    entry = (
        atom_names=atom_names,
        atom_ref_coords=atom_ref_coords,
        bonds=bonds,
    )
    _ccd_entry_cache[key] = entry
    return entry
end
