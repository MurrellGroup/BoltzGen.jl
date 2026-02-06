function _canonical_element_symbol_smiles(sym::AbstractString)
    s = strip(String(sym))
    isempty(s) && return "C"
    chars = collect(lowercase(s))
    chars[1] = uppercase(chars[1])
    return String(chars)
end

function _next_atom_name_smiles(sym::String, counts::Dict{String, Int})
    n = get(counts, sym, 0) + 1
    counts[sym] = n
    name = string(sym, n)
    ncodeunits(name) <= 4 && return name
    short = string(sym[1], n)
    return ncodeunits(short) <= 4 ? short : string(sym[1], mod(n, 999))
end

function _coords_from_molecule_smiles(mol, n_atoms::Int)
    coords3 = get(mol.props, :coordinates_3d, nothing)
    if coords3 !== nothing && size(coords3, 1) == n_atoms && size(coords3, 2) >= 3
        return Float32.(coords3[:, 1:3])
    end

    coords2 = get(mol.props, :coordinates_2d, nothing)
    if coords2 !== nothing && size(coords2, 1) == n_atoms && size(coords2, 2) >= 2
        out = zeros(Float32, n_atoms, 3)
        out[:, 1] .= Float32.(coords2[:, 1])
        out[:, 2] .= Float32.(coords2[:, 2])
        return out
    end

    out = zeros(Float32, n_atoms, 3)
    for i in 1:n_atoms
        out[i, 1] = 1.5f0 * (i - 1)
    end
    return out
end

function _smiles_to_ligand_tokens_moleculeflow(smiles_list::Vector{String})
    tokens = String[]
    token_atom_names = Vector{Vector{String}}()
    token_atom_coords = Vector{Dict{String,NTuple{3,Float32}}}()

    for smiles in smiles_list
        s = String(strip(smiles))
        isempty(s) && error("SMILES string is empty")

        mol = MoleculeFlow.mol_from_smiles(s)
        mol.valid || error("Invalid SMILES: $s")

        confs = MoleculeFlow.generate_3d_conformers(mol, 1; random_seed=1)
        mol_conf = if !isempty(confs)
            confs[1].molecule
        else
            confs2 = MoleculeFlow.generate_2d_conformers(mol)
            isempty(confs2) ? mol : confs2[1].molecule
        end

        atoms_any = MoleculeFlow.get_atoms(mol_conf)
        atoms_any === missing && error("Failed to extract atoms from SMILES: $s")
        atoms = atoms_any::Vector
        n_atoms = length(atoms)
        n_atoms > 0 || error("No atoms found in SMILES: $s")
        coords = _coords_from_molecule_smiles(mol_conf, n_atoms)

        counts = Dict{String, Int}()
        names = String[]
        xyz = Dict{String, NTuple{3, Float32}}()
        for i in 1:n_atoms
            sym = _canonical_element_symbol_smiles(MoleculeFlow.get_symbol(atoms[i]))
            atom_name = _next_atom_name_smiles(sym, counts)
            push!(names, atom_name)
            xyz[atom_name] = (
                Float32(coords[i, 1]),
                Float32(coords[i, 2]),
                Float32(coords[i, 3]),
            )
        end

        push!(tokens, "UNK")
        push!(token_atom_names, names)
        push!(token_atom_coords, xyz)
    end

    return (
        tokens=tokens,
        token_atom_names=token_atom_names,
        token_atom_coords=token_atom_coords,
    )
end

function smiles_to_ligand_tokens(smiles_list::Vector{String})
    isempty(smiles_list) && error("SMILES list is empty")
    for s in smiles_list
        isempty(strip(s)) && error("SMILES string is empty")
    end

    _HAS_MOLECULEFLOW || error(
        "SMILES ligand parsing requires optional dependency MoleculeFlow.jl. " *
        "Add `MoleculeFlow` to your environment and re-load BoltzGen to enable ligand.smiles support.",
    )
    return _smiles_to_ligand_tokens_moleculeflow(smiles_list)
end
const _HAS_MOLECULEFLOW = let
    try
        @eval import MoleculeFlow
        true
    catch
        false
    end
end
