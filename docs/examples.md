# Examples & Workflows

Detailed usage examples for BoltzGen.jl, organized by workflow.

## Setup

```julia
using BoltzGen

# For GPU support:
using CUDA, cuDNN
```

Model weights are downloaded automatically from HuggingFace on first use and cached locally.

---

## Design Workflows (BoltzGen1)

All design workflows use the BoltzGen1 model:

```julia
gen = BoltzGen.load_boltzgen()          # CPU
gen = BoltzGen.load_boltzgen(; gpu=true)  # GPU
```

### De Novo Protein Design

Generate a completely new protein structure from scratch.

```julia
# Simplest path — just specify length
result = BoltzGen.denovo_sample(gen, 30; steps=100, seed=7)

# Equivalent via design_from_sequence with empty sequence
result = BoltzGen.design_from_sequence(gen, ""; length=30, steps=100, seed=7)

# Write output files
BoltzGen.write_outputs(result, "denovo_30mer")

# Or get PDB as a string (useful for piping to visualization tools)
pdb = BoltzGen.output_to_pdb(result)
```

### Sequence Redesign

Fix some positions and let the model redesign others:

```julia
# 10-residue scaffold with positions 3-6 designable
seq = "GGGGGGGGGG"
mask = [false, false, true, true, true, true, false, false, false, false]
result = BoltzGen.design_from_sequence(gen, seq; design_mask=mask, steps=100, seed=42)

BoltzGen.write_outputs(result, "redesigned")
```

### Cyclic Peptide Design

```julia
result = BoltzGen.design_from_sequence(gen, ""; length=8, cyclic=true, steps=100, seed=1)
BoltzGen.write_outputs(result, "cyclic_8mer")
```

### Design with Small Molecules (YAML)

YAML specification files describe multi-entity complexes for design:

```julia
# Protein binding a CCD ligand (chorismite)
result = BoltzGen.design_from_yaml(gen,
    "examples/protein_binding_small_molecule/chorismite.yaml";
    steps=100, seed=7)

# Mixed protein + DNA + SMILES ligand
result = BoltzGen.design_from_yaml(gen,
    "examples/mixed_protein_dna_smiles_smoke_v1.yaml";
    steps=100, seed=7)

BoltzGen.write_outputs(result, "mixed_complex")
```

### Target-Conditioned Design

Design a new chain that binds to an existing target structure:

```julia
# Design a 20-residue binder for a target protein
result = BoltzGen.target_conditioned_design(gen, "target.pdb";
    design_length=20, steps=100, seed=42)

# Restrict to specific target chains
result = BoltzGen.target_conditioned_design(gen, "complex.pdb";
    design_length=15, include_chains=["A", "B"], steps=100, seed=42)

BoltzGen.write_outputs(result, "binder")
```

### Using the Alternative Weights

BoltzGen1 has two checkpoint variants:

```julia
# Default: diverse sampling
gen_diverse = BoltzGen.load_boltzgen()

# Alternative: higher sequence adherence
gen_adhere = BoltzGen.load_boltzgen(; weights="boltzgen1_adherence_state_dict.safetensors")
```

---

## Folding Workflows (Boltz2)

All folding workflows use the Boltz2 model:

```julia
fold = BoltzGen.load_boltz2()                    # confidence only
fold = BoltzGen.load_boltz2(; gpu=true)          # confidence on GPU
aff  = BoltzGen.load_boltz2(; affinity=true)     # confidence + affinity
```

### Fold a Single Chain

Predict 3D structure from an amino acid sequence:

```julia
fold = BoltzGen.load_boltz2()

result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY";
    steps=200, recycles=3, seed=7)

# Confidence metrics
metrics = BoltzGen.confidence_metrics(result)
println("pTM: $(metrics.ptm)")
println("ipTM: $(metrics.iptm)")
println("pLDDT: $(metrics.complex_plddt)")

BoltzGen.write_outputs(result, "fold_20mer")
```

### Fold an Antibody

```julia
fold = BoltzGen.load_boltz2(; gpu=true)

# Trastuzumab VH (Herceptin, 121 residues)
vh = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"

result = BoltzGen.fold_from_sequence(fold, vh; steps=200, recycles=3, seed=42)

# Validate geometry
stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"])
println("Atoms: $(stats.n_atoms), max_abs: $(round(stats.max_abs; digits=1))")

# Confidence
metrics = BoltzGen.confidence_metrics(result)
println("pTM: $(round(metrics.ptm[1]; digits=3))")

BoltzGen.write_outputs(result, "trastuzumab_vh")
```

### Fold a Multi-Chain Complex

```julia
fold = BoltzGen.load_boltz2()

chain_A = "ACDEFGHIKLMNPQRSTVWY"
chain_B = "FGHIKLMNPQRSTVWYACDE"
result = BoltzGen.fold_from_sequences(fold, [chain_A, chain_B]; steps=200, seed=7)

metrics = BoltzGen.confidence_metrics(result)
println("ipTM (interface quality): $(metrics.iptm)")

BoltzGen.write_outputs(result, "complex_AB")
```

### Re-fold a Structure

Refine or re-predict structure from an existing PDB/CIF file:

```julia
fold = BoltzGen.load_boltz2()

result = BoltzGen.fold_from_structure(fold, "input.pdb"; steps=100, seed=7)

# Restrict to specific chains
result = BoltzGen.fold_from_structure(fold, "complex.pdb";
    include_chains=["A"], steps=100, seed=7)

BoltzGen.write_outputs(result, "refolded")
```

### Affinity Prediction

Predict binding affinity for a protein-ligand complex:

```julia
aff = BoltzGen.load_boltz2(; affinity=true, gpu=true)

result = BoltzGen.fold_from_structure(aff, "complex.pdb";
    include_nonpolymer=true, steps=100, seed=7)

metrics = BoltzGen.confidence_metrics(result)
println("Affinity: $(metrics.affinity_pred_value)")
println("Binding probability: $(metrics.affinity_probability_binary)")

# Multiple affinity head outputs are available:
# metrics.affinity_pred_value, metrics.affinity_pred_value1, metrics.affinity_pred_value2
# metrics.affinity_probability_binary, metrics.affinity_probability_binary1, metrics.affinity_probability_binary2
```

### Fold with MSA

All fold functions accept pre-computed MSAs via `msa_file` (path to FASTA/A3M) or `msa_sequences` (Vector{String}). BoltzGen.jl does **not** perform MSA search — you must provide alignments from external tools (e.g., MMseqs2, HHblits, Jackhmmer).

```julia
fold = BoltzGen.load_boltz2()

# From a file
result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY";
    msa_file="my_alignment.a3m", steps=200, seed=7)

# From pre-loaded sequences
msa = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWF", "ACDEYGHIKLMNPQRSTVWY"]
result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY";
    msa_sequences=msa, max_msa_rows=1024, steps=200, seed=7)
```

### Fold with Structural Templates

Provide known structures as templates to guide folding:

```julia
fold = BoltzGen.load_boltz2()

result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY";
    template_paths=["homolog1.cif", "homolog2.pdb"],
    max_templates=4, steps=200, seed=7)
```

Template features populated: `template_ca`, `template_cb`, `template_frame_rot`, `template_frame_t`, and `template_mask`.

### What Boltz2 Does NOT Support

- **Design/generative sampling**: Design masks are rejected. Use BoltzGen1 for protein design.
- **Automatic MSA search**: You must bring your own pre-computed alignments.

---

## Output Handling

### Output Formats

BoltzGen.jl produces three output formats:

| Format | Function | Description |
|--------|----------|-------------|
| PDB atom14 | `output_to_pdb(result)` | PDB with internal atom14 representation |
| PDB atom37 | `output_to_pdb_atom37(result)` | PDB with standard atom37 ordering |
| mmCIF | `output_to_mmcif(result)` | mmCIF format |

### Writing Files

```julia
# Write all three formats at once
BoltzGen.write_outputs(result, "output/my_protein")
# Creates:
#   output/my_protein_atom14.pdb
#   output/my_protein_atom37.pdb
#   output/my_protein_atom37.cif
```

### Getting Strings

```julia
# Get output as strings (no files written)
pdb14 = BoltzGen.output_to_pdb(result)
pdb37 = BoltzGen.output_to_pdb_atom37(result)
cif   = BoltzGen.output_to_mmcif(result)

# Useful for piping to other tools or embedding in notebooks
```

### Batched Results

For batched inference, specify the batch index:

```julia
pdb = BoltzGen.output_to_pdb(result; batch=2)
BoltzGen.write_outputs(result, "batch2"; batch=2)
```

---

## Geometry Validation

### Sanity Check

```julia
# Throws on failure; returns stats on success
stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"])

println("Atoms: $(stats.n_atoms)")
println("Max absolute coordinate: $(stats.max_abs)")
println("Min nearest neighbor: $(stats.min_nearest_neighbor)")
println("Max nearest neighbor: $(stats.max_nearest_neighbor)")
println("Fraction with nearest < 0.5 A: $(stats.frac_nearest_lt_0p5)")
println("Fraction with |coord| >= 900: $(stats.frac_abs_ge900)")
```

### Bond Length Validation

```julia
violations = BoltzGen.check_bond_lengths(result)

if isempty(violations)
    println("All bonds within expected ranges")
else
    BoltzGen.print_bond_length_report(violations)
end
```

---

## YAML Specification Format

Design inputs can be specified as YAML files describing multi-entity complexes.

### Protein Only (De Novo)

```yaml
entities:
  - protein:
      id: A
      sequence: 20          # integer = de novo design of this length
```

### Protein with Length Range

```yaml
entities:
  - protein:
      id: A
      sequence: 140..180    # length range for de novo design
```

### Protein + CCD Ligand

```yaml
entities:
  - protein:
      id: A
      sequence: 140..180
  - ligand:
      id: B
      ccd: TSA              # CCD component code (e.g., TSA for chorismite)
```

### Protein + DNA + SMILES Ligand

```yaml
entities:
  - protein:
      id: A
      sequence: 12
  - dna:
      id: B
      sequence: ACGTACGT
  - ligand:
      id: C
      smiles: "CC(=O)OC1=CC=CC=C1C(=O)O"
```

### Supported Entity Types

| Type | Fields | Description |
|------|--------|-------------|
| `protein` | `id`, `sequence` | Protein chain. `sequence` can be an amino acid string, integer (de novo length), or range (e.g., `140..180`) |
| `dna` | `id`, `sequence` | DNA chain with nucleotide sequence |
| `rna` | `id`, `sequence` | RNA chain with nucleotide sequence |
| `ligand` | `id`, `smiles` or `ccd` | Small molecule. Specify either a SMILES string or a CCD component code |

### Example YAML Files

The repository includes example YAML inputs:

- `examples/protein_binding_small_molecule/chorismite.yaml` — protein + CCD ligand
- `examples/mixed_protein_dna_smiles_smoke_v1.yaml` — protein + DNA + SMILES
- `examples/e2e_tests/` — 8 end-to-end test inputs covering various entity combinations
- `examples/parity_tests/` — 35 parity validation inputs

---

## Complete Workflow: Design-then-Fold

A typical workflow combines design (BoltzGen1) with structure validation (Boltz2):

```julia
using BoltzGen

# Step 1: Design a protein
gen = BoltzGen.load_boltzgen(; gpu=true)
design = BoltzGen.design_from_sequence(gen, ""; length=50, steps=200, seed=42)
BoltzGen.write_outputs(design, "designed_50mer")

# Step 2: Fold the designed sequence to validate
fold = BoltzGen.load_boltz2(; gpu=true)
# (Extract the designed sequence from the result, or fold from the output file)
fold_result = BoltzGen.fold_from_structure(fold, "designed_50mer_atom37.pdb"; steps=200, seed=7)

# Step 3: Evaluate confidence
metrics = BoltzGen.confidence_metrics(fold_result)
println("Designed protein quality:")
println("  pTM:   $(round(metrics.ptm[1]; digits=3))")
println("  pLDDT: $(round(metrics.complex_plddt[1]; digits=3))")

# Step 4: Validate geometry
stats = BoltzGen.assert_geometry_sane_atom37!(fold_result["feats"], fold_result["coords"])
violations = BoltzGen.check_bond_lengths(fold_result)
BoltzGen.print_bond_length_report(violations)
```

**Note**: When re-loading generated PDB files containing small molecules, be aware of the [ligand tokenization limitation](status.md#2-fold-from-structure-ligand-tokenization). Prefer starting from original YAML/SMILES inputs when possible.
