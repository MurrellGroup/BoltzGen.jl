module BoltzGenMoleculeFlowExt

using BoltzGen
using MoleculeFlow

function BoltzGen._smiles_to_ligand_tokens_moleculeflow(smiles_list::Vector{String})
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
        coords = BoltzGen._coords_from_molecule_smiles(mol_conf, n_atoms)

        counts = Dict{String, Int}()
        names = String[]
        xyz = Dict{String, NTuple{3, Float32}}()
        for i in 1:n_atoms
            sym = BoltzGen._canonical_element_symbol_smiles(MoleculeFlow.get_symbol(atoms[i]))
            atom_name = BoltzGen._next_atom_name_smiles(sym, counts)
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
                bt = BoltzGen._smiles_bond_type_token_id(MoleculeFlow.get_bond_type(bnd))
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

end
