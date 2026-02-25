# Developer Guide

Notes for contributors and developers working on BoltzGen.jl.

## Source Layout

```
src/
  BoltzGen.jl                — Module definition and exports
  api.jl                     — REPL API (BoltzGenHandle, load/design/fold functions)
  boltz.jl                   — BoltzModel struct and forward pass
  checkpoint.jl              — Weight loading and HuggingFace resolution
  const.jl                   — Constants (token IDs, chain types, atom types from ProtInterop)
  features.jl                — Feature tensor construction (design, fold, refinement paths)
  output.jl                  — PDB/mmCIF output generation and geometry checks
  diffusion.jl               — Diffusion module and sampling
  diffusion_conditioning.jl  — Diffusion conditioning
  encoders.jl                — Atom encoder/decoder
  trunk.jl                   — Pairformer trunk, template module
  confidence.jl              — Confidence heads (pTM, ipTM, pLDDT)
  affinity.jl                — Affinity prediction heads
  transformers.jl            — Transformer layers (tri-axial attention)
  masker.jl                  — Feature masking for inference
  utils.jl                   — Coordinate utilities, augmentation helpers
  yaml_parser.jl             — YAML design specification parser
  smiles.jl                  — SMILES handling
  ccd_cache.jl               — CCD molecule cache (JLD2)
  bond_lengths.jl            — Bond length validation
```

## Architecture Overview

### Model Families

BoltzGen.jl contains two model families sharing most of their architecture:

- **BoltzGen1** — generative design model. Uses diffusion sampling with design masks to generate protein structures and sequences.
- **Boltz2** — structure prediction (folding) model. Uses the same core architecture but with confidence and affinity heads, and without design mask support.

Both are wrapped by `BoltzGenHandle`, which stores the loaded model, its family, and capability flags.

### Data Flow

1. **Input processing** — Sequences/YAML/structure files are tokenized and converted to feature tensors (`features.jl`, `yaml_parser.jl`)
2. **Feature masking** — Design masks applied for generative tasks (`masker.jl`)
3. **Forward pass** — Features flow through atom encoder → pairformer trunk → diffusion sampling → atom decoder (`boltz.jl`, `trunk.jl`, `diffusion.jl`, `encoders.jl`)
4. **Post-processing** — Raw coordinates are post-processed and optionally run through confidence/affinity heads (`output.jl`, `confidence.jl`, `affinity.jl`)
5. **Output** — Results packaged as a Dict and optionally written to PDB/mmCIF files (`api.jl`, `output.jl`)

### GPU Patterns

Key patterns used throughout the codebase for GPU compatibility:

- **Device-aware allocation**: `Onion.zeros_like(ref)`, `fill!(similar(ref, T, dims), val)`, `copyto!(similar(ref, ...), cpu_array)` — tensors are allocated on the same device as a reference tensor.
- **No scalar indexing**: All operations use vectorized broadcasting, `NNlib.batched_mul`, or equivalent. Scalar indexing into GPU arrays will error.
- **CPU boundary for confidence/affinity**: These modules run on CPU after moving data from GPU. This is a deliberate choice for numerical stability.
- **Transfer pattern**: Features are transferred to GPU before the forward pass; results are moved back to CPU afterward. The user always receives CPU results.

### CCD Molecule Data

Small-molecule design cases (SMILES, CCD ligands) require the CCD molecule cache — a JLD2 file containing pre-processed chemical component dictionary entries. This is downloaded automatically from HuggingFace on first use.

The cache path can be overridden via the `BOLTZGEN_MOLS_JLD2` environment variable.

### Weight Resolution

`resolve_weights_path()` in `checkpoint.jl` handles weight file resolution:

1. Check known HuggingFace aliases (e.g., `"boltzgen1_diverse"` → `"boltzgen1_diverse_state_dict.safetensors"`)
2. Download from the HuggingFace repo (`MurrellLab/BoltzGen.jl`) on first use
3. Cache locally for subsequent runs

## Dependencies

| Package | Role |
|---------|------|
| `Onion.jl` | ML framework (layers, GPU utilities, parameter management) |
| `ProtInterop.jl` | Shared protein/nucleic acid utilities (tokenization, structure parsing, bond checking) |
| `HuggingFaceApi.jl` | Automatic weight downloading |
| `JLD2` | CCD molecule cache format |
| `SafeTensors` | Model weight file format |
| `NNlib` | Neural network primitives (batched_mul, etc.) |
| `YAML` | YAML specification parsing |
| `MoleculeFlow` | (optional extension) Small-molecule SMILES handling |

## Testing

### Comprehensive Test Suite

The single-session test script runs all 8 API test cases in one Julia session, avoiding per-script recompilation:

```bash
julia --project=<env> BoltzGen.jl/scripts/run_repl_api_tests.jl         # CPU
julia --project=<env> BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu   # GPU
julia --project=<env> BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu /path/to/outputs
```

### Test Cases

| Case | API Function | Model | Description |
|------|-------------|-------|-------------|
| case01 | `design_from_sequence` | BoltzGen1 | De novo protein design (12 residues) |
| case02 | `design_from_yaml` | BoltzGen1 | Protein + small molecule (CCD ligand) |
| case03 | `design_from_yaml` | BoltzGen1 | Protein + DNA + SMILES (mixed complex) |
| case07 | `target_conditioned_design` | BoltzGen1 | Design conditioned on case01 output |
| case04 | `fold_from_sequence` | Boltz2 conf | Fold poly-glycine, verify confidence |
| antibody | `fold_from_sequence` | Boltz2 conf | Fold trastuzumab VH (121 residues) |
| case05 | `fold_from_sequence` | Boltz2 aff | Fold poly-glycine, verify affinity |
| case06 | `fold_from_structure` | Boltz2 aff | Re-fold case03 output with affinity |

Each case validates:
- Geometry sanity via `assert_geometry_sane_atom37!`
- String output generation (PDB atom14, PDB atom37, mmCIF)
- File output writing
- Confidence/affinity metrics where applicable

### Prerequisites

- BoltzGen.jl and its dependencies (Onion.jl, ProtInterop.jl)
- A Julia project environment with all dependencies resolved
- SMILES cases (case03) require the `MoleculeFlow` extension
- Model weights are downloaded from HuggingFace on first use
- GPU tests require `CUDA.jl` and `cuDNN.jl`

### Legacy Scripts

Individual scripts in `scripts/` can still be run standalone. Each requires its own Julia process and command-line arguments. The REPL API test suite is preferred as it runs in a single session (faster compilation) and tests the programmatic API directly.

## YAML Parity Tests

The `examples/parity_tests/` directory contains 35 YAML inputs used for validating feature-level parity against the Python reference implementation. These are used by dedicated parity sweep scripts rather than the standard test suite.

## Extension: MoleculeFlow

The `BoltzGenMoleculeFlowExt` extension (defined in `Project.toml`) is loaded automatically when `MoleculeFlow` is available. It provides SMILES parsing and small-molecule featurization needed for ligand-containing design cases.
