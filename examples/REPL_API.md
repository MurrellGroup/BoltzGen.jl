# BoltzGen.jl REPL API

A Julia API for protein design and structure prediction using BoltzGen1 (design) and Boltz2 (folding) models. Load a model once, run multiple generations interactively, and retrieve results as dictionaries.

## Quick Start

```julia
# Activate project and load BoltzGen
using BoltzGen

# Load a design model (downloads weights from HuggingFace on first use)
gen = BoltzGen.load_boltzgen()

# Design a 20-residue protein de novo
result = BoltzGen.design_from_sequence(gen, ""; length=20, steps=100, recycles=3, seed=42)

# Write output files
BoltzGen.write_outputs(result, "my_design")
# Creates: my_design_atom14.pdb, my_design_atom37.pdb, my_design_atom37.cif

# Get output as strings (no files written)
pdb_string = BoltzGen.output_to_pdb(result)
```

## GPU Support

Pass `gpu=true` when loading a model to run on GPU. Requires CUDA.jl and cuDNN.jl.

```julia
using CUDA, cuDNN

gen = BoltzGen.load_boltzgen(; gpu=true)
result = BoltzGen.design_from_sequence(gen, ""; length=20, steps=100, seed=42)
# Results are always returned on CPU; GPU is used only for the forward pass.
```

## Model Loading

### `load_boltzgen(; weights, gpu) -> BoltzGenHandle`

Load a BoltzGen1 design model for protein design tasks.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `weights` | `String` | `"boltzgen1_diverse_state_dict.safetensors"` | Weight file name or local path |
| `gpu` | `Bool` | `false` | Move model to GPU |

```julia
gen = BoltzGen.load_boltzgen()                           # default (diverse)
gen = BoltzGen.load_boltzgen(; gpu=true)                 # GPU mode
gen = BoltzGen.load_boltzgen(; weights="boltzgen1_adherence_state_dict.safetensors")
```

### `load_boltz2(; affinity, weights, gpu) -> BoltzGenHandle`

Load a Boltz2 folding model for structure prediction with confidence scoring.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `affinity` | `Bool` | `false` | Load affinity head (uses boltz2_aff checkpoint) |
| `weights` | `String` | auto | Weight file; auto-selects based on `affinity` flag |
| `gpu` | `Bool` | `false` | Move model to GPU |

```julia
fold = BoltzGen.load_boltz2()                  # confidence only
fold = BoltzGen.load_boltz2(; gpu=true)        # confidence on GPU
aff  = BoltzGen.load_boltz2(; affinity=true)   # confidence + affinity
```

### `BoltzGenHandle`

The opaque model handle returned by load functions. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `model_family` | `String` | `"boltzgen1"` or `"boltz2"` |
| `weights_name` | `String` | Basename of loaded checkpoint |
| `has_confidence` | `Bool` | Whether confidence heads are available |
| `has_affinity` | `Bool` | Whether affinity heads are available |
| `on_gpu` | `Bool` | Whether model is on GPU |

```julia
julia> gen
BoltzGenHandle(boltzgen1, weights="boltzgen1_diverse_state_dict.safetensors")

julia> fold
BoltzGenHandle(boltz2, weights="boltz2_conf_final_state_dict.safetensors", confidence, gpu)
```

## Design Functions

These require a BoltzGen1 handle (from `load_boltzgen()`).

### `design_from_sequence(handle, sequence; ...) -> Dict`

Design a protein from a sequence or de novo. Equivalent to the `run_design_from_sequence.jl` script.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | BoltzGen1 model |
| `sequence` | `String` | `""` | Amino acid sequence (e.g. `"ACDEFG"`); empty for de novo |
| `length` | `Int` | `0` | Residue count for de novo design (when `sequence=""`) |
| `chain_type` | `String` | `"PROTEIN"` | Chain type: `"PROTEIN"`, `"DNA"`, or `"RNA"` |
| `design_mask` | `Vector{Bool}` | auto | Per-residue mask: `true` = designable. Auto: all-true for de novo, all-false for fixed sequence |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Number of recycling iterations |
| `seed` | `Int` | `nothing` | Random seed for reproducibility |
| `cyclic` | `Bool` | `false` | Enforce cyclic peptide topology |

```julia
# De novo design (12 residues)
result = BoltzGen.design_from_sequence(gen, ""; length=12, steps=20, recycles=2, seed=7)

# Redesign specific positions in a fixed scaffold
seq = "GGGGGGGGGG"
mask = [false, false, true, true, true, true, false, false, false, false]
result = BoltzGen.design_from_sequence(gen, seq; design_mask=mask, steps=100, seed=42)

# Design a cyclic peptide
result = BoltzGen.design_from_sequence(gen, ""; length=8, cyclic=true, steps=100, seed=1)
```

### `design_from_yaml(handle, yaml_path; ...) -> Dict`

Design from a YAML specification file. Supports protein, DNA, RNA, small molecules (SMILES), and mixed complexes. Equivalent to the `run_design_from_yaml.jl` script.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | BoltzGen1 model |
| `yaml_path` | `String` | required | Path to YAML design specification |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |
| `include_nonpolymer` | `Bool` | `true` | Include small molecules from YAML |

```julia
result = BoltzGen.design_from_yaml(gen, "examples/protein_binding_small_molecule/chorismite.yaml";
    steps=20, recycles=2, seed=7)
```

### `target_conditioned_design(handle, target_path; ...) -> Dict`

Design a new chain conditioned on an existing target structure. Equivalent to `run_target_conditioned_design.jl`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | BoltzGen1 model |
| `target_path` | `String` | required | PDB/CIF file with target structure |
| `design_length` | `Int` | `8` | Number of residues in designed chain |
| `design_chain_type` | `String` | `"PROTEIN"` | Type of designed chain: `"PROTEIN"`, `"DNA"`, `"RNA"` |
| `include_chains` | `Vector{String}` | `nothing` | Restrict to specific target chains |
| `include_nonpolymer` | `Bool` | `false` | Include ligands from target |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |

```julia
result = BoltzGen.target_conditioned_design(gen, "target.pdb";
    design_length=20, steps=100, recycles=3, seed=42)
```

### `denovo_sample(handle, length; ...) -> Dict`

Simplest de novo sampling path using `build_denovo_atom14_features`. Equivalent to `run_denovo_sample.jl`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | BoltzGen1 model |
| `length` | `Int` | required | Number of residues |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |

```julia
result = BoltzGen.denovo_sample(gen, 30; steps=100, seed=7)
```

## Fold Functions

These require a Boltz2 handle (from `load_boltz2()`).

### `fold_from_sequence(handle, sequence; ...) -> Dict`

Predict the 3D structure of a protein from its amino acid sequence. Returns coordinates and confidence metrics. Equivalent to `run_fold_from_sequence.jl`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | Boltz2 model |
| `sequence` | `String` | required | Amino acid sequence |
| `chain_type` | `String` | `"PROTEIN"` | Chain type |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |

```julia
fold = BoltzGen.load_boltz2()
result = BoltzGen.fold_from_sequence(fold, "EVQLVESGGGLVQPGGSLRLSC"; steps=200, recycles=3, seed=7)
metrics = BoltzGen.confidence_metrics(result)
println("pTM: ", metrics.ptm, "  ipTM: ", metrics.iptm)
```

### `fold_from_structure(handle, target_path; ...) -> Dict`

Re-fold/refine a structure from a PDB/CIF file. Useful for confidence scoring of existing structures or affinity prediction. Equivalent to `run_fold_from_structure.jl`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | Boltz2 model |
| `target_path` | `String` | required | Input PDB/CIF file |
| `include_chains` | `Vector{String}` | `nothing` | Restrict to specific chains |
| `include_nonpolymer` | `Bool` | `true` | Include ligands |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |

```julia
aff = BoltzGen.load_boltz2(; affinity=true)
result = BoltzGen.fold_from_structure(aff, "complex.pdb"; steps=100, seed=7)
metrics = BoltzGen.confidence_metrics(result)
println("Affinity: ", metrics.affinity_pred_value)
```

## Output Functions

### `output_to_pdb(result; batch=1) -> String`

Generate a PDB-format string from the atom14 representation.

### `output_to_pdb_atom37(result; batch=1) -> String`

Generate a PDB-format string with standard atom37 atom ordering.

### `output_to_mmcif(result; batch=1) -> String`

Generate an mmCIF-format string.

### `write_outputs(result, prefix; batch=1)`

Write all three output files at once:
- `{prefix}_atom14.pdb`
- `{prefix}_atom37.pdb`
- `{prefix}_atom37.cif`

```julia
BoltzGen.write_outputs(result, "output/my_protein")
# Creates: output/my_protein_atom14.pdb, output/my_protein_atom37.pdb, output/my_protein_atom37.cif
```

### `confidence_metrics(result) -> NamedTuple`

Extract confidence and affinity metrics from a fold result. Returns a NamedTuple with only the keys present in the result.

**Confidence model keys:** `ptm`, `iptm`, `complex_plddt`, `complex_iplddt`

**Affinity model keys (additional):** `affinity_pred_value`, `affinity_probability_binary`, `affinity_pred_value1`, `affinity_probability_binary1`, `affinity_pred_value2`, `affinity_probability_binary2`

```julia
metrics = BoltzGen.confidence_metrics(result)
metrics.ptm              # predicted TM-score
metrics.iptm             # interface predicted TM-score
metrics.complex_plddt    # predicted per-residue LDDT (complex-level)
```

## Result Dictionary

All API functions return a `Dict{String,Any}` with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `"feats"` | `Dict{String,Any}` | Post-processed features (atom names, residue types, etc.) |
| `"feats_orig"` | `Dict{String,Any}` | Original features before masking/design |
| `"coords"` | `Array{Float32}` | Atom coordinates, shape `(3, M)` or `(3, M, B)` |
| `"sample_atom_coords"` | `Array{Float32}` | Raw sampled coordinates from diffusion |
| `"pdistogram"` | `Array{Float32}` | Pairwise distance histogram |
| `"ptm"`, `"iptm"`, ... | `Array{Float32}` | Confidence/affinity head outputs (Boltz2 only) |

## Geometry Validation

```julia
stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"]; batch=1)
# Throws on failure. Returns a NamedTuple of geometry statistics on success:
#   n_atoms, max_abs, min_nearest_neighbor, max_nearest_neighbor,
#   frac_nearest_lt_0p5, frac_abs_ge900
```

## Complete Example: Antibody VH Folding on GPU

```julia
using CUDA, cuDNN
using BoltzGen

# Load Boltz2 confidence model on GPU
fold = BoltzGen.load_boltz2(; gpu=true)

# Trastuzumab VH sequence (Herceptin, 121 residues)
vh_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"

# Fold with 200 diffusion steps
result = BoltzGen.fold_from_sequence(fold, vh_seq; steps=200, recycles=3, seed=42)

# Check geometry
stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"])
println("Atoms: $(stats.n_atoms), max_abs: $(round(stats.max_abs; digits=1))")

# Get confidence
metrics = BoltzGen.confidence_metrics(result)
println("pTM: $(round(metrics.ptm[1]; digits=3))")

# Write outputs
BoltzGen.write_outputs(result, "trastuzumab_vh")

# Get PDB as string (for piping to other tools)
pdb = BoltzGen.output_to_pdb_atom37(result)
```

## Running the Test Suite

A comprehensive single-session test script covers all functionality:

```bash
# CPU
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl

# GPU
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu

# Custom output directory
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu /path/to/outputs
```

The test script runs 8 cases covering all API functions with geometry validation:

| Case | Function | Description |
|------|----------|-------------|
| case01 | `design_from_sequence` | De novo protein design (12 residues) |
| case02 | `design_from_yaml` | Protein + small molecule (chorismite) |
| case03 | `design_from_yaml` | Protein + DNA + SMILES (mixed complex) |
| case07 | `target_conditioned_design` | Design conditioned on case01 output |
| case04 | `fold_from_sequence` | Fold poly-glycine with confidence |
| antibody | `fold_from_sequence` | Fold trastuzumab VH (121 residues) |
| case05 | `fold_from_sequence` | Fold poly-glycine with affinity |
| case06 | `fold_from_structure` | Re-fold case03 structure with affinity |

## Weight Files

Weights are automatically downloaded from HuggingFace on first use. Available checkpoints:

| Name | File | Use |
|------|------|-----|
| BoltzGen1 Diverse | `boltzgen1_diverse_state_dict.safetensors` | Default design model |
| BoltzGen1 Adherence | `boltzgen1_adherence_state_dict.safetensors` | Alternative design model |
| Boltz2 Confidence | `boltz2_conf_final_state_dict.safetensors` | Default fold model |
| Boltz2 Affinity | `boltz2_aff_state_dict.safetensors` | Fold + affinity model |

Weight resolution order:
1. If the path is a local file that exists, use it directly
2. Otherwise, download from HuggingFace (`MurrellLab/BoltzGen.jl` repo) and cache
