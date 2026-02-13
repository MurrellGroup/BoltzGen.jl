# Developer Notes

Last updated: 2026-02-13

## 1) Workspace Layout

This workspace is at `/home/claudey/ProteinModels/WorkingBoltzGen/`.

Primary folders:
- `BoltzGen.jl/` — BoltzGen Julia implementation (branch: `gpu`)
- `Onion.jl/` — ML framework dependency (branch: `bg_optimized`)
- `runfromhere/` — Julia project environment for running everything

## 2) Runtime Rules

```bash
julia --project=runfromhere BoltzGen.jl/scripts/<script>.jl <args...>
```

Constraints:
- **Never run two Julia processes simultaneously** (especially on GPU — will kill the machine)
- **Never modify Julia memory limits**
- No silent fallbacks — errors must surface immediately
- Keep strict behavior in core code

## 3) REPL API

The primary interface is the REPL-friendly API. See `examples/REPL_API.md` for full documentation.

Quick overview:
```julia
using BoltzGen

# Design
gen = BoltzGen.load_boltzgen(; gpu=true)
result = BoltzGen.design_from_sequence(gen, ""; length=20, steps=100, seed=42)
BoltzGen.write_outputs(result, "my_design")

# Fold
fold = BoltzGen.load_boltz2(; gpu=true)
result = BoltzGen.fold_from_sequence(fold, "ACDEFGHIKLMNPQRSTVWY"; steps=200, seed=7)
metrics = BoltzGen.confidence_metrics(result)
```

## 4) Testing

Single-session test covering all 8 cases:
```bash
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl        # CPU
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu  # GPU
```

Expected: 8 PASS, 0 FAIL on both CPU and GPU.

## 5) Weight Files

Resolved via `resolve_weights_path()` in `src/checkpoint.jl`:
1. If path is a local file, use directly
2. Otherwise download from HuggingFace (`MurrellLab/BoltzGen.jl`)

Available checkpoints:

| File | Use |
|------|-----|
| `boltzgen1_diverse_state_dict.safetensors` | Default design model |
| `boltzgen1_adherence_state_dict.safetensors` | Alternative design model |
| `boltz2_conf_final_state_dict.safetensors` | Default fold model (confidence) |
| `boltz2_aff_state_dict.safetensors` | Fold model with affinity |

## 6) Source Layout

```
BoltzGen.jl/src/
  BoltzGen.jl    — Module definition and exports
  api.jl         — REPL API (BoltzGenHandle, load/design/fold functions, output functions)
  boltz.jl       — BoltzModel struct and forward pass
  checkpoint.jl  — Weight loading and resolution
  const.jl       — Constants (token IDs, chain types, atom types)
  features.jl    — Feature construction (build_design_features, etc.)
  output.jl      — PDB/mmCIF output (write_pdb, write_pdb_atom37, write_mmcif, geometry checks)
  diffusion.jl   — Diffusion module and sampling
  diffusion_conditioning.jl — Diffusion conditioning
  encoders.jl    — Atom encoder/decoder
  trunk.jl       — Pairformer trunk, template module
  confidence.jl  — Confidence heads
  affinity.jl    — Affinity heads
  transformers.jl — Transformer layers
  masker.jl      — Feature masking for inference
  utils.jl       — Coordinate utilities, augmentation
  yaml_parser.jl — YAML design specification parser
  smiles.jl      — SMILES handling
  ccd_cache.jl   — CCD molecule cache
```

## 7) GPU Architecture

Key patterns used for GPU compatibility:
- **Device-aware allocation**: `Onion.zeros_like(ref)`, `fill!(similar(ref, T, dims), val)`, `copyto!(similar(ref, ...), cpu_array)`
- **No scalar indexing**: All operations use vectorized broadcasting, `NNlib.batched_mul`, or `Einops.rearrange`
- **CPU boundary for confidence/affinity**: These modules run on CPU after moving data from GPU
- **Forward pass**: Features transferred to GPU before forward, results moved back to CPU after

## 8) CCD Molecule Data

Required for SMILES/small-molecule design cases (case02, case03). The JLD2 cache is downloaded from HuggingFace on first use.

Override cache path via `BOLTZGEN_MOLS_JLD2` environment variable.

## 9) Known Limitations

- YAML parser parity with Python is incomplete for some edge cases
- Fold-from-structure can collapse ligand tokenization when reloading generated PDB files
- Confidence/affinity modules always run on CPU
- Boltz2 is fold-only in current guards (no design with Boltz2)
