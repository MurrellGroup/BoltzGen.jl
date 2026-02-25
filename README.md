# BoltzGen.jl

A Julia implementation of the BoltzGen/Boltz2 protein design and structure prediction pipeline. Supports both **protein design** (BoltzGen1) and **structure prediction/folding** (Boltz2) with a REPL-friendly API for interactive use.

## Features

- **Protein design** — de novo design, sequence redesign, target-conditioned design, YAML-driven mixed-complex design
- **Structure prediction** — fold proteins from sequence, re-fold/refine from structure files
- **Confidence scoring** — pTM, ipTM, pLDDT metrics via Boltz2 confidence heads
- **Binding affinity prediction** — predict binding affinity for protein-ligand complexes
- **MSA and template support** — pass pre-computed MSAs (FASTA/A3M) and structural templates to improve predictions
- **Mixed-entity support** — protein, DNA, RNA, small molecules (SMILES and CCD codes)
- **GPU acceleration** — full GPU inference via CUDA.jl; results always returned on CPU
- **Multiple output formats** — PDB (atom14 and atom37 representations) and mmCIF
- **Geometry validation** — built-in bond length checking and structure sanity assertions
- **Automatic weight management** — model weights downloaded from HuggingFace on first use

## Installation

BoltzGen.jl is not yet registered in the Julia General registry. Add it directly from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/MurrellGroup/BoltzGen.jl", rev="consolidation")
```

### Dependencies

BoltzGen.jl depends on [Onion.jl](https://github.com/MurrellGroup/Onion.jl) (ML framework) and [ProtInterop.jl](https://github.com/MurrellGroup/ProtInterop.jl) (shared protein utilities). These must be available in your environment.

For GPU support, you also need:
```julia
Pkg.add(["CUDA", "cuDNN"])
```

For small-molecule support (SMILES inputs), the optional `MoleculeFlow` extension is loaded automatically when `MoleculeFlow` is available in your environment.

## Quick Start

```julia
using BoltzGen
```

### Design a protein de novo (BoltzGen1)

```julia
# Load the BoltzGen1 design model (downloads weights on first use)
gen = BoltzGen.load_boltzgen()

# Design a 20-residue protein
result = BoltzGen.design_from_sequence(gen, ""; length=20, steps=100, seed=42)

# Write PDB and mmCIF output files
BoltzGen.write_outputs(result, "my_design")
# Creates: my_design_atom14.pdb, my_design_atom37.pdb, my_design_atom37.cif
```

### Fold a protein from sequence (Boltz2)

```julia
# Load the Boltz2 folding model
fold = BoltzGen.load_boltz2()

# Predict structure from amino acid sequence
result = BoltzGen.fold_from_sequence(fold, "EVQLVESGGGLVQPGGSLRLSC"; steps=200, seed=7)

# Get confidence metrics
metrics = BoltzGen.confidence_metrics(result)
println("pTM: ", metrics.ptm, "  ipTM: ", metrics.iptm)

# Write output
BoltzGen.write_outputs(result, "my_fold")
```

### GPU acceleration

```julia
using CUDA, cuDNN

gen = BoltzGen.load_boltzgen(; gpu=true)
result = BoltzGen.design_from_sequence(gen, ""; length=20, steps=100, seed=42)
# Results always returned on CPU — GPU used only during forward pass
```

## Models

BoltzGen.jl includes two model families:

### BoltzGen1 — Protein Design

BoltzGen1 is a generative model for protein structure design. It supports:
- De novo protein design (generate new proteins from scratch)
- Sequence redesign (redesign specific positions in a scaffold)
- Cyclic peptide design
- Target-conditioned design (design a binder for a given target structure)
- Mixed-complex design from YAML specifications (protein + DNA/RNA + small molecules)

Load with `BoltzGen.load_boltzgen()`.

### Boltz2 — Structure Prediction (Folding)

Boltz2 is a structure prediction model. It supports:
- **Fold from sequence** — predict 3D structure from amino acid sequence (`fold_from_sequence`)
- **Multi-chain folding** — fold multi-chain complexes from multiple sequences (`fold_from_sequences`)
- **Fold from structure** — re-fold or refine an existing PDB/CIF structure (`fold_from_structure`)
- **Confidence scoring** — pTM, ipTM, complex pLDDT, complex ipLDDT metrics
- **Binding affinity prediction** — predict affinity for protein-ligand complexes (with `affinity=true`)

**What Boltz2 does NOT support in this package:**
- Design/generative sampling (design masks are rejected — use BoltzGen1 for design)
- Automatic MSA search (you must provide pre-computed MSAs; the package does not run search tools like MMseqs2 or HHblits)

Load with `BoltzGen.load_boltz2()`.

## REPL Examples

### De novo design

```julia
gen = BoltzGen.load_boltzgen()

# Simplest path: denovo_sample
result = BoltzGen.denovo_sample(gen, 30; steps=100, seed=7)

# Or via design_from_sequence with empty sequence
result = BoltzGen.design_from_sequence(gen, ""; length=30, steps=100, seed=7)

# Get PDB as a string (no files written)
pdb_str = BoltzGen.output_to_pdb(result)
```

### Redesign specific positions

```julia
gen = BoltzGen.load_boltzgen()

seq = "GGGGGGGGGG"
mask = [false, false, true, true, true, true, false, false, false, false]
result = BoltzGen.design_from_sequence(gen, seq; design_mask=mask, steps=100, seed=42)
```

### Cyclic peptide design

```julia
gen = BoltzGen.load_boltzgen()
result = BoltzGen.design_from_sequence(gen, ""; length=8, cyclic=true, steps=100, seed=1)
```

### Design with small molecules (YAML)

```julia
gen = BoltzGen.load_boltzgen()

# Protein binding a CCD ligand
result = BoltzGen.design_from_yaml(gen,
    "examples/protein_binding_small_molecule/chorismite.yaml";
    steps=20, seed=7)

# Mixed protein + DNA + SMILES ligand
result = BoltzGen.design_from_yaml(gen,
    "examples/mixed_protein_dna_smiles_smoke_v1.yaml";
    steps=20, seed=7)
```

### Target-conditioned design

```julia
gen = BoltzGen.load_boltzgen()

result = BoltzGen.target_conditioned_design(gen, "target.pdb";
    design_length=20, steps=100, seed=42)
```

### Fold a protein sequence (Boltz2)

```julia
fold = BoltzGen.load_boltz2()

# Fold a short sequence
result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY"; steps=200, seed=7)
metrics = BoltzGen.confidence_metrics(result)
println("pTM: $(metrics.ptm)  pLDDT: $(metrics.complex_plddt)")

# Write outputs
BoltzGen.write_outputs(result, "fold_result")
```

### Fold with MSA and templates

All fold (and design) functions accept pre-computed MSAs and structural templates:

```julia
fold = BoltzGen.load_boltz2()

# Fold with a pre-computed MSA (FASTA/A3M file)
result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY";
    msa_file="my_msa.a3m", steps=200, seed=7)

# Or pass MSA sequences directly
msa = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWF", "ACDEFGHIKLMNPQRSTVWH"]
result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY";
    msa_sequences=msa, max_msa_rows=1024, steps=200, seed=7)

# Fold with structural templates
result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY";
    template_paths=["template1.cif", "template2.pdb"], steps=200, seed=7)
```

Note: BoltzGen.jl does **not** perform MSA search — you must provide pre-computed alignments from external tools (e.g., MMseqs2, HHblits, Jackhmmer).

### Fold an antibody VH domain

```julia
fold = BoltzGen.load_boltz2(; gpu=true)

# Trastuzumab VH (Herceptin, 121 residues)
vh = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
result = BoltzGen.fold_from_sequence(fold, vh; steps=200, recycles=3, seed=42)

stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"])
metrics = BoltzGen.confidence_metrics(result)
println("pTM: $(round(metrics.ptm[1]; digits=3))")

BoltzGen.write_outputs(result, "trastuzumab_vh")
```

### Fold a multi-chain complex

```julia
fold = BoltzGen.load_boltz2()

chain_A = "ACDEFGHIKLMNPQRSTVWY"
chain_B = "FGHIKLMNPQRSTVWYACDE"
result = BoltzGen.fold_from_sequences(fold, [chain_A, chain_B]; steps=200, seed=7)
```

### Re-fold a structure with affinity prediction

```julia
aff = BoltzGen.load_boltz2(; affinity=true)

result = BoltzGen.fold_from_structure(aff, "complex.pdb"; steps=100, seed=7)
metrics = BoltzGen.confidence_metrics(result)
println("Affinity: ", metrics.affinity_pred_value)
println("Binding probability: ", metrics.affinity_probability_binary)
```

### Geometry validation

```julia
# Assert geometry is sane (throws on failure)
stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"])
println("Atoms: $(stats.n_atoms), min_nn: $(stats.min_nearest_neighbor)")

# Check bond lengths
violations = BoltzGen.check_bond_lengths(result)
BoltzGen.print_bond_length_report(violations)
```

### Output formats

```julia
# Get output as strings
pdb14 = BoltzGen.output_to_pdb(result)           # atom14 PDB
pdb37 = BoltzGen.output_to_pdb_atom37(result)     # atom37 PDB
cif   = BoltzGen.output_to_mmcif(result)           # mmCIF

# Write all three files at once
BoltzGen.write_outputs(result, "output/my_protein")
# Creates: output/my_protein_atom14.pdb
#          output/my_protein_atom37.pdb
#          output/my_protein_atom37.cif
```

## YAML Specification Format

Design inputs can be specified as YAML files describing multi-entity complexes:

```yaml
# Protein binding a CCD ligand
entities:
  - protein:
      id: A
      sequence: 140..180    # length range for de novo design
  - ligand:
      id: B
      ccd: TSA              # CCD component code
```

```yaml
# Mixed protein + DNA + SMILES ligand
entities:
  - protein:
      id: A
      sequence: 12           # fixed length for de novo design
  - dna:
      id: B
      sequence: ACGTACGT
  - ligand:
      id: C
      smiles: "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Supported entity types: `protein`, `dna`, `rna`, `ligand` (with `smiles` or `ccd` field).

## Available Checkpoints

Model weights are automatically downloaded from HuggingFace (`MurrellLab/BoltzGen.jl`) on first use and cached locally.

| Checkpoint | File | Use |
|-----------|------|-----|
| BoltzGen1 Diverse | `boltzgen1_diverse_state_dict.safetensors` | Default design model |
| BoltzGen1 Adherence | `boltzgen1_adherence_state_dict.safetensors` | Alternative design model (higher sequence adherence) |
| Boltz2 Confidence | `boltz2_conf_final_state_dict.safetensors` | Default fold model with confidence heads |
| Boltz2 Affinity | `boltz2_aff_state_dict.safetensors` | Fold model with confidence + affinity heads |

## Result Dictionary

All API functions return a `Dict{String, Any}`:

| Key | Type | Description |
|-----|------|-------------|
| `"feats"` | `Dict{String,Any}` | Post-processed features (atom names, residue types, chain IDs, etc.) |
| `"feats_orig"` | `Dict{String,Any}` | Original features before masking/design |
| `"coords"` | `Array{Float32}` | Atom coordinates — shape `(3, M)` or `(3, M, B)` for batched |
| `"sample_atom_coords"` | `Array{Float32}` | Raw sampled coordinates from diffusion |
| `"pdistogram"` | `Array{Float32}` | Pairwise distance histogram |
| `"ptm"`, `"iptm"`, ... | `Array{Float32}` | Confidence metrics (Boltz2 only) |
| `"affinity_pred_value"`, ... | `Array{Float32}` | Affinity metrics (Boltz2 affinity model only) |

## Documentation

- **[API Reference](docs/api.md)** — complete function signatures, parameters, and return types
- **[Examples](docs/examples.md)** — detailed usage examples, workflows, and YAML format reference
- **[Developer Guide](docs/developer.md)** — source layout, GPU patterns, testing, runtime constraints
- **[Status & Known Issues](docs/status.md)** — feature parity, known limitations, fix priorities

## Testing

Run the comprehensive test suite (8 cases covering all API functions):

```bash
# CPU
julia --project=<env> BoltzGen.jl/scripts/run_repl_api_tests.jl

# GPU
julia --project=<env> BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu
```

Expected output:
```
REPL API TEST SUMMARY
  PASS  case01_design_protein_only
  PASS  case02_design_protein_small_molecule
  PASS  case03_design_protein_dna_smiles
  PASS  case07_target_conditioned_design
  PASS  case04_fold_sequence_confidence
  PASS  case_antibody_fold
  PASS  case05_fold_sequence_affinity
  PASS  case06_fold_structure_affinity
  Total: 8  Pass: 8  Fail: 0
```

## License

See repository for license information.
