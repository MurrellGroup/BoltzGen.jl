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

function _smiles_bond_type_token_id(bt_any)
    bt = uppercase(strip(string(bt_any)))
    bt0 = get(bond_type_ids, bt, bond_type_ids["OTHER"])
    # Python tokenization stores bond type ids offset by +1 (0 reserved for no-bond).
    return Int(bt0 + 1)
end

function _coords_from_molecule_smiles(mol, n_atoms::Int)
    coords3 = get(mol.props, :coordinates_3d, nothing)
    coords3 === nothing && error("SMILES conformer is missing :coordinates_3d")
    size(coords3, 2) >= 3 || error("SMILES conformer has invalid coordinate shape: $(size(coords3))")
    if size(coords3, 1) == n_atoms
        return Float32.(coords3[:, 1:3])
    elseif size(coords3, 1) > n_atoms
        # MoleculeFlow can attach H-expanded conformer coordinates while
        # `get_atoms` returns heavy atoms only. Keep heavy-atom rows.
        return Float32.(coords3[1:n_atoms, 1:3])
    end
    error("SMILES conformer atom count mismatch: coords=$(size(coords3, 1)) atoms=$n_atoms")
end

function _smiles_to_ligand_tokens_moleculeflow(smiles_list::Vector{String})
    tokens = String[]
    token_atom_names = Vector{Vector{String}}()
    token_atom_coords = Vector{Dict{String,NTuple{3,Float32}}}()
    token_bonds = Vector{Vector{NTuple{3,Int}}}()

    for smiles in smiles_list
        s = String(strip(smiles))
        isempty(s) && error("SMILES string is empty")

        mol = MoleculeFlow.mol_from_smiles(s)
        mol.valid || error("Invalid SMILES: $s")

        confs = MoleculeFlow.generate_3d_conformers(mol, 1; random_seed=1)
        isempty(confs) && error("Failed to generate 3D conformer for SMILES: $s")
        mol_conf = confs[1].molecule

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

        edge_types = Dict{Tuple{Int,Int},Int}()
        for i in 1:n_atoms
            bonds_any = MoleculeFlow.get_bonds_from_atom(mol_conf, i)
            bonds_any === missing && continue
            bonds = bonds_any::Vector
            for bnd in bonds
                a = Int(MoleculeFlow.get_begin_atom_idx(bnd))
                b = Int(MoleculeFlow.get_end_atom_idx(bnd))
                1 <= a <= n_atoms || error("SMILES bond begin index out of range: $a (n_atoms=$n_atoms)")
                1 <= b <= n_atoms || error("SMILES bond end index out of range: $b (n_atoms=$n_atoms)")
                a == b && continue
                i1, i2 = a < b ? (a, b) : (b, a)
                bt = _smiles_bond_type_token_id(MoleculeFlow.get_bond_type(bnd))
                if haskey(edge_types, (i1, i2))
                    edge_types[(i1, i2)] == bt || error("SMILES bond type mismatch for edge ($i1,$i2): $(edge_types[(i1,i2)]) vs $bt")
                else
                    edge_types[(i1, i2)] = bt
                end
            end
        end
        local_bonds = NTuple{3,Int}[]
        for (edge, bt) in sort(collect(edge_types); by=x -> (x[1][1], x[1][2]))
            push!(local_bonds, (edge[1], edge[2], bt))
        end

        push!(tokens, "UNK")
        push!(token_atom_names, names)
        push!(token_atom_coords, xyz)
        push!(token_bonds, local_bonds)
    end

    return (
        tokens=tokens,
        token_atom_names=token_atom_names,
        token_atom_coords=token_atom_coords,
        token_bonds=token_bonds,
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
