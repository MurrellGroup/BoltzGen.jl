const chain_types = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
const chain_type_ids = Dict(chain => i-1 for (i, chain) in enumerate(chain_types))

const canonical_tokens = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

const non_canonical_tokens = [
    "<pad>",
    "-",
    "UNK",
    "A",
    "G",
    "C",
    "U",
    "N",
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",
]

const canonicals_offset = 2
const tokens = vcat(non_canonical_tokens[1:canonicals_offset], canonical_tokens, non_canonical_tokens[canonicals_offset+1:end])
const num_tokens = length(tokens)
const token_ids = Dict(token => i for (i, token) in enumerate(tokens))
const token_ids0 = Dict(token => i-1 for (i, token) in enumerate(tokens))
const unk_token = Dict("PROTEIN" => "UNK", "DNA" => "DN", "RNA" => "N")

const prot_letter_to_token = Dict(
    'A' => "ALA",
    'R' => "ARG",
    'N' => "ASN",
    'D' => "ASP",
    'C' => "CYS",
    'E' => "GLU",
    'Q' => "GLN",
    'G' => "GLY",
    'H' => "HIS",
    'I' => "ILE",
    'L' => "LEU",
    'K' => "LYS",
    'M' => "MET",
    'F' => "PHE",
    'P' => "PRO",
    'S' => "SER",
    'T' => "THR",
    'W' => "TRP",
    'Y' => "TYR",
    'V' => "VAL",
    'X' => "UNK",
    'J' => "UNK",
    'B' => "UNK",
    'Z' => "UNK",
    'O' => "UNK",
    'U' => "UNK",
    '-' => "-",
)

const rna_letter_to_token = Dict(
    'A' => "A",
    'G' => "G",
    'C' => "C",
    'U' => "U",
    'N' => "N",
)

const dna_letter_to_token = Dict(
    'A' => "DA",
    'G' => "DG",
    'C' => "DC",
    'T' => "DT",
    'N' => "DN",
)

const bond_types = [
    "OTHER",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "COVALENT",
]
const bond_type_ids = Dict(bond => i-1 for (i, bond) in enumerate(bond_types))

const binding_types = [
    "UNSPECIFIED",
    "BINDING",
    "NOT_BINDING",
]
const binding_type_ids = Dict(binding => i-1 for (i, binding) in enumerate(binding_types))

const ss_types = [
    "UNSPECIFIED",
    "LOOP",
    "HELIX",
    "SHEET",
]
const ss_type_ids = Dict(ss => i-1 for (i, ss) in enumerate(ss_types))

const contact_conditioning_info = Dict(
    "UNSPECIFIED" => 0,
    "UNSELECTED" => 1,
    "POCKET>BINDER" => 2,
    "BINDER>POCKET" => 3,
    "CONTACT" => 4,
)

const chunk_size_threshold = 512

const num_method_types = 12
const num_temp_bins = 4
const num_ph_bins = 4

const atom_types = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
const atom_order = Dict(atom => i for (i, atom) in enumerate(atom_types))

const ref_atoms = Dict(
    "UNK" => ["N", "CA", "C", "O", "CB"],
    "GLY" => ["N", "CA", "C", "O"],
    "ALA" => ["N", "CA", "C", "O", "CB"],
    "CYS" => ["N", "CA", "C", "O", "CB", "SG"],
    "SER" => ["N", "CA", "C", "O", "CB", "OG"],
    "PRO" => ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "THR" => ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "VAL" => ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "ASN" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "ILE" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "MET" => ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "GLN" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "LYS" => ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "HIS" => ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "PHE" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "ARG" => ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "TYR" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "TRP" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "A" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "N" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],
    "DA" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "DG" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "DC" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "DT" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
    "DN" => ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"],
)

const protein_backbone_atom_names = Set(["N", "CA", "C", "O"])
const nucleic_backbone_atom_names = Set(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"])

const res_to_center_atom = Dict(
    "UNK" => "CA", "ALA" => "CA", "ARG" => "CA", "ASN" => "CA", "ASP" => "CA",
    "CYS" => "CA", "GLN" => "CA", "GLU" => "CA", "GLY" => "CA", "HIS" => "CA",
    "ILE" => "CA", "LEU" => "CA", "LYS" => "CA", "MET" => "CA", "PHE" => "CA",
    "PRO" => "CA", "SER" => "CA", "THR" => "CA", "TRP" => "CA", "TYR" => "CA", "VAL" => "CA",
    "A" => "C1'", "G" => "C1'", "C" => "C1'", "U" => "C1'", "N" => "C1'",
    "DA" => "C1'", "DG" => "C1'", "DC" => "C1'", "DT" => "C1'", "DN" => "C1'",
)

const res_to_disto_atom = Dict(
    "UNK" => "CB", "ALA" => "CB", "ARG" => "CB", "ASN" => "CB", "ASP" => "CB",
    "CYS" => "CB", "GLN" => "CB", "GLU" => "CB", "GLY" => "CA", "HIS" => "CB",
    "ILE" => "CB", "LEU" => "CB", "LYS" => "CB", "MET" => "CB", "PHE" => "CB",
    "PRO" => "CB", "SER" => "CB", "THR" => "CB", "TRP" => "CB", "TYR" => "CB", "VAL" => "CB",
    "A" => "C4", "G" => "C4", "C" => "C2", "U" => "C2", "N" => "C1'",
    "DA" => "C4", "DG" => "C4", "DC" => "C2", "DT" => "C2", "DN" => "C1'",
)

const res_to_center_atom_id = Dict(res => findfirst(==(atom), ref_atoms[res]) for (res, atom) in res_to_center_atom if haskey(ref_atoms, res))
const res_to_disto_atom_id = Dict(res => findfirst(==(atom), ref_atoms[res]) for (res, atom) in res_to_disto_atom if haskey(ref_atoms, res))

# Element info (for masking and one-hot element channels)
const num_elements = 128
const mask_element_id = 114 # atomic number for FL in boltzgen.data.const
const mask_element_index = mask_element_id + 1

include("ref_pos_table.jl")
include("ref_charge_table.jl")
