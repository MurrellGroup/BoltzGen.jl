# ── Aliases from ProtInterop (Phase 1 consolidation) ──────────────────────────
# These were previously defined here; now sourced from the shared package.

const chain_types = ProtInterop.CHAIN_TYPES
const chain_type_ids = ProtInterop.CHAIN_TYPE_IDS
const canonical_tokens = ProtInterop.CANONICAL_AMINO_ACIDS
const prot_letter_to_token = ProtInterop.PROTEIN_LETTER_TO_TOKEN
const rna_letter_to_token = ProtInterop.RNA_LETTER_TO_TOKEN
const dna_letter_to_token = ProtInterop.DNA_LETTER_TO_TOKEN
const atom_types = ProtInterop.PROTEIN_ATOM_TYPES
const atom_order = ProtInterop.PROTEIN_ATOM_ORDER
const ref_atoms = ProtInterop.RESIDUE_ATOMS
const protein_backbone_atom_names = ProtInterop.PROTEIN_BACKBONE_ATOMS
const nucleic_backbone_atom_names = ProtInterop.NUCLEIC_BACKBONE_ATOMS
const res_to_center_atom = ProtInterop.RESIDUE_CENTER_ATOM
const res_to_disto_atom = ProtInterop.RESIDUE_DISTO_ATOM
const res_to_center_atom_id = ProtInterop.RESIDUE_CENTER_ATOM_ID
const res_to_disto_atom_id = ProtInterop.RESIDUE_DISTO_ATOM_ID
# ref_atom_pos and ref_atom_charge are exported by ProtInterop and available via `using`

# ── Model-specific token vocabulary (BoltzGen ordering) ──────────────────────

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

# ── Bond / binding / secondary structure types ────────────────────────────────

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

# ── Model hyperparameters ─────────────────────────────────────────────────────

const chunk_size_threshold = 512

const num_method_types = 12
const num_temp_bins = 4
const num_ph_bins = 4

# Element info (for masking and one-hot element channels)
const num_elements = 128
const mask_element_id = 114 # atomic number for FL in boltzgen.data.const
const mask_element_index = mask_element_id + 1
