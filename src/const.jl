# Minimal constants ported from boltzgen.data.const for inference

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

# Element info (for masking)
const num_elements = 128
const mask_element_id = 114 # atomic number for FL in boltzgen.data.const
const mask_element_index = mask_element_id + 1
