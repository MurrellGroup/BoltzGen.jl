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

# Stub: real implementation provided by BoltzGenMoleculeFlowExt when MoleculeFlow is loaded.
function _smiles_to_ligand_tokens_moleculeflow end

function smiles_to_ligand_tokens(smiles_list::Vector{String})
    isempty(smiles_list) && error("SMILES list is empty")
    for s in smiles_list
        isempty(strip(s)) && error("SMILES string is empty")
    end

    if !hasmethod(_smiles_to_ligand_tokens_moleculeflow, Tuple{Vector{String}})
        error(
            "SMILES ligand parsing requires optional dependency MoleculeFlow.jl. " *
            "Add `MoleculeFlow` to your environment and load it before BoltzGen.",
        )
    end
    return _smiles_to_ligand_tokens_moleculeflow(smiles_list)
end
