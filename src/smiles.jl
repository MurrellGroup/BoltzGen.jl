function smiles_to_ligand_tokens(smiles_list::Vector{String})
    isempty(smiles_list) && error("SMILES list is empty")
    for s in smiles_list
        isempty(strip(s)) && error("SMILES string is empty")
    end
    error(
        "SMILES ligand parsing requires optional dependency MoleculeFlow.jl. " *
        "Add `MoleculeFlow` to your environment and re-load BoltzGen to enable ligand.smiles support.",
    )
end

