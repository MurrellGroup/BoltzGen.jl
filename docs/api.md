# API Reference

Complete reference for the BoltzGen.jl public API.

## Model Loading

### `load_boltzgen(; kwargs...) -> BoltzGenHandle`

Load a BoltzGen1 design model for protein design tasks.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `weights` | `String` | `"boltzgen1_diverse_state_dict.safetensors"` | Weight file name or local path |
| `gpu` | `Bool` | `false` | Move model to GPU |

```julia
gen = BoltzGen.load_boltzgen()                                                    # default (diverse)
gen = BoltzGen.load_boltzgen(; gpu=true)                                          # GPU mode
gen = BoltzGen.load_boltzgen(; weights="boltzgen1_adherence_state_dict.safetensors")  # adherence model
```

### `load_boltz2(; kwargs...) -> BoltzGenHandle`

Load a Boltz2 folding model for structure prediction with confidence scoring.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `affinity` | `Bool` | `false` | Load affinity head (uses `boltz2_aff` checkpoint) |
| `weights` | `String` | auto | Weight file; auto-selects based on `affinity` flag |
| `gpu` | `Bool` | `false` | Move model to GPU |

```julia
fold = BoltzGen.load_boltz2()                   # confidence only
fold = BoltzGen.load_boltz2(; gpu=true)         # confidence on GPU
aff  = BoltzGen.load_boltz2(; affinity=true)    # confidence + affinity
```

### `BoltzGenHandle`

The opaque model handle returned by load functions.

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

---

## Design Functions (BoltzGen1)

These require a BoltzGen1 handle from `load_boltzgen()`.

### `design_from_sequence(handle, sequence; kwargs...) -> Dict`

Design a protein from a sequence or de novo.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | BoltzGen1 model |
| `sequence` | `String` | `""` | Amino acid sequence; empty string for de novo |
| `length` | `Int` | `0` | Residue count for de novo (when `sequence=""`) |
| `chain_type` | `String` | `"PROTEIN"` | `"PROTEIN"`, `"DNA"`, or `"RNA"` |
| `design_mask` | `Vector{Bool}` | auto | Per-residue mask: `true` = designable |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed for reproducibility |
| `cyclic` | `Bool` | `false` | Enforce cyclic peptide topology |

```julia
# De novo design (12 residues)
result = BoltzGen.design_from_sequence(gen, ""; length=12, steps=20, seed=7)

# Redesign specific positions
seq = "GGGGGGGGGG"
mask = [false, false, true, true, true, true, false, false, false, false]
result = BoltzGen.design_from_sequence(gen, seq; design_mask=mask, steps=100, seed=42)

# Cyclic peptide
result = BoltzGen.design_from_sequence(gen, ""; length=8, cyclic=true, steps=100, seed=1)
```

### `design_from_yaml(handle, yaml_path; kwargs...) -> Dict`

Design from a YAML specification file. Supports protein, DNA, RNA, small molecules (SMILES/CCD), and mixed complexes.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | BoltzGen1 model |
| `yaml_path` | `String` | required | Path to YAML design specification |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |
| `include_nonpolymer` | `Bool` | `true` | Include small molecules from YAML |

```julia
result = BoltzGen.design_from_yaml(gen,
    "examples/protein_binding_small_molecule/chorismite.yaml";
    steps=20, seed=7)
```

### `target_conditioned_design(handle, target_path; kwargs...) -> Dict`

Design a new chain conditioned on an existing target structure.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | BoltzGen1 model |
| `target_path` | `String` | required | PDB/CIF file with target structure |
| `design_length` | `Int` | `8` | Number of residues in designed chain |
| `design_chain_type` | `String` | `"PROTEIN"` | `"PROTEIN"`, `"DNA"`, or `"RNA"` |
| `include_chains` | `Vector{String}` | `nothing` | Restrict to specific target chains |
| `include_nonpolymer` | `Bool` | `false` | Include ligands from target |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |

```julia
result = BoltzGen.target_conditioned_design(gen, "target.pdb";
    design_length=20, steps=100, seed=42)
```

### `denovo_sample(handle, length; kwargs...) -> Dict`

Simplest de novo sampling path. Thin wrapper around `build_denovo_atom14_features`.

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

---

## Fold Functions (Boltz2)

These require a Boltz2 handle from `load_boltz2()`.

**Boltz2 supports folding/structure prediction only.** Design masks are rejected — use BoltzGen1 for generative design.

### `fold_from_sequence(handle, sequence; kwargs...) -> Dict`

Predict the 3D structure of a protein from its amino acid sequence. Returns coordinates, confidence metrics, and (optionally) affinity predictions.

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
result = BoltzGen.fold_from_sequence(fold, "EVQLVESGGGLVQPGGSLRLSC"; steps=200, seed=7)
metrics = BoltzGen.confidence_metrics(result)
println("pTM: ", metrics.ptm, "  ipTM: ", metrics.iptm)
```

### `fold_from_sequences(handle, sequences; kwargs...) -> Dict`

Fold a multi-chain complex from multiple sequences.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | Boltz2 model |
| `sequences` | `Vector{String}` | required | Amino acid sequences (one per chain) |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |

```julia
fold = BoltzGen.load_boltz2()
result = BoltzGen.fold_from_sequences(fold, ["ACDEFGHIK", "LMNPQRSTVWY"]; steps=200, seed=7)
```

### `fold_from_structure(handle, target_path; kwargs...) -> Dict`

Re-fold or refine a structure from a PDB/CIF file. Useful for confidence scoring of existing structures or affinity prediction.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `handle` | `BoltzGenHandle` | required | Boltz2 model |
| `target_path` | `String` | required | Input PDB/CIF file |
| `include_chains` | `Vector{String}` | `nothing` | Restrict to specific chains |
| `include_nonpolymer` | `Bool` | `true` | Include ligands from structure |
| `steps` | `Int` | `100` | Diffusion sampling steps |
| `recycles` | `Int` | `3` | Recycling iterations |
| `seed` | `Int` | `nothing` | Random seed |

```julia
aff = BoltzGen.load_boltz2(; affinity=true)
result = BoltzGen.fold_from_structure(aff, "complex.pdb"; steps=100, seed=7)
metrics = BoltzGen.confidence_metrics(result)
println("Affinity: ", metrics.affinity_pred_value)
```

---

## Output Functions

### `output_to_pdb(result; batch=1) -> String`

Generate a PDB-format string from the atom14 representation.

### `output_to_pdb_atom37(result; batch=1) -> String`

Generate a PDB-format string with standard atom37 atom ordering.

### `output_to_mmcif(result; batch=1) -> String`

Generate an mmCIF-format string.

### `write_outputs(result, prefix; batch=1)`

Write all three output formats at once:
- `{prefix}_atom14.pdb`
- `{prefix}_atom37.pdb`
- `{prefix}_atom37.cif`

```julia
BoltzGen.write_outputs(result, "output/my_protein")
```

### `confidence_metrics(result) -> NamedTuple`

Extract confidence and affinity metrics from a fold result. Returns a NamedTuple containing only the keys present in the result dictionary.

**Confidence model keys:** `ptm`, `iptm`, `complex_plddt`, `complex_iplddt`

**Affinity model keys (additional):** `affinity_pred_value`, `affinity_probability_binary`, `affinity_pred_value1`, `affinity_probability_binary1`, `affinity_pred_value2`, `affinity_probability_binary2`

```julia
metrics = BoltzGen.confidence_metrics(result)
metrics.ptm              # predicted TM-score
metrics.iptm             # interface predicted TM-score
metrics.complex_plddt    # predicted per-residue LDDT (complex-level)
```

---

## Geometry Validation

### `assert_geometry_sane_atom37!(feats, coords; batch=1) -> NamedTuple`

Assert that output geometry is physically reasonable. Throws an error on failure.

Returns a NamedTuple with fields: `n_atoms`, `max_abs`, `min_nearest_neighbor`, `max_nearest_neighbor`, `frac_nearest_lt_0p5`, `frac_abs_ge900`.

```julia
stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"])
println("Atoms: $(stats.n_atoms), min_nn: $(stats.min_nearest_neighbor)")
```

### `check_bond_lengths(result) -> Vector{BondViolation}`

Check bond lengths in the output structure. Returns a vector of `BondViolation` objects for any bonds outside expected ranges.

### `print_bond_length_report(violations)`

Print a formatted report of bond length violations.

```julia
violations = BoltzGen.check_bond_lengths(result)
BoltzGen.print_bond_length_report(violations)
```

---

## Result Dictionary

All API functions return a `Dict{String, Any}` with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `"feats"` | `Dict{String,Any}` | Post-processed features (atom names, residue types, chain IDs, etc.) |
| `"feats_orig"` | `Dict{String,Any}` | Original features before masking/design |
| `"coords"` | `Array{Float32}` | Atom coordinates — shape `(3, M)` or `(3, M, B)` |
| `"sample_atom_coords"` | `Array{Float32}` | Raw sampled coordinates from diffusion |
| `"pdistogram"` | `Array{Float32}` | Pairwise distance histogram |
| `"ptm"`, `"iptm"`, ... | `Array{Float32}` | Confidence head outputs (Boltz2 only) |
| `"affinity_pred_value"`, ... | `Array{Float32}` | Affinity outputs (Boltz2 affinity model only) |

---

## Lower-Level API

These functions are exported for advanced use but most users should prefer the REPL API above.

### Feature Construction

| Function | Description |
|----------|-------------|
| `build_denovo_atom14_features(N)` | Build features for de novo design of `N` residues |
| `build_denovo_atom14_features_from_sequence(seq)` | Build features from a sequence string |
| `build_design_features(yaml_data)` | Build features from parsed YAML data |
| `build_design_features_from_structure(tokens)` | Build features from structure tokens |

### Parsing & Tokenization

| Function | Description |
|----------|-------------|
| `tokens_from_sequence(seq)` | Tokenize an amino acid sequence |
| `load_structure_tokens(path; kwargs...)` | Load and tokenize a PDB/CIF structure |
| `parse_design_yaml(path)` | Parse a YAML design specification |
| `load_msa_sequences(path)` | Load MSA sequences from a file (re-exported from ProtInterop) |

### Model Operations

| Function | Description |
|----------|-------------|
| `boltz_forward(model, feats; kwargs...)` | Run model forward pass |
| `BoltzModel` | Model struct type |
| `infer_config(state_dict)` | Infer model config from weight shapes |
| `load_params!(model, state_dict)` | Load parameters into a model |
| `load_model_from_state(state_dict; kwargs...)` | Build and load model from state dict |
| `load_model_from_safetensors(path; kwargs...)` | Build and load model from safetensors file |
| `resolve_weights_path(name)` | Resolve weight file name to local path (downloads if needed) |
| `default_weights_filename(family)` | Get default weights filename for a model family |
| `boltz_masker(feats; kwargs...)` | Apply inference masking to features |

### Output (Low-Level)

| Function | Description |
|----------|-------------|
| `write_pdb(io, feats, coords; kwargs...)` | Write PDB to IO stream |
| `write_pdb_atom37(io, feats, coords; kwargs...)` | Write atom37 PDB to IO stream |
| `write_mmcif(io, feats, coords; kwargs...)` | Write mmCIF to IO stream |
| `collect_atom37_entries(feats, coords; kwargs...)` | Collect atom37 entries as structured data |
| `postprocess_atom14(feats, coords)` | Post-process atom14 features and coordinates |
| `geometry_stats_atom37(feats, coords; kwargs...)` | Compute geometry statistics |

---

## Available Checkpoints

| Checkpoint | File | Use |
|-----------|------|-----|
| BoltzGen1 Diverse | `boltzgen1_diverse_state_dict.safetensors` | Default design model |
| BoltzGen1 Adherence | `boltzgen1_adherence_state_dict.safetensors` | Alternative design model |
| Boltz2 Confidence | `boltz2_conf_final_state_dict.safetensors` | Default fold model |
| Boltz2 Affinity | `boltz2_aff_state_dict.safetensors` | Fold + affinity model |

Weights are downloaded from HuggingFace (`MurrellLab/BoltzGen.jl`) on first use and cached locally. Weight resolution is handled by `resolve_weights_path()`.
