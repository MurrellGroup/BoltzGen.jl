# Comprehensive Tests

## Recommended: Single-Session REPL API Tests

The single-session test script `run_repl_api_tests.jl` runs all 8 test cases in one Julia session,
avoiding per-script recompilation overhead. Models are loaded once and reused.

```bash
# CPU
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl

# GPU
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu

# Custom output directory
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu /path/to/outputs
```

### Test Cases

| Case | API Function | Model | Description |
|------|-------------|-------|-------------|
| case01 | `design_from_sequence` | BoltzGen1 | De novo protein design (12 residues) |
| case02 | `design_from_yaml` | BoltzGen1 | Protein + small molecule (chorismite) |
| case03 | `design_from_yaml` | BoltzGen1 | Protein + DNA + SMILES mixed complex |
| case07 | `target_conditioned_design` | BoltzGen1 | Design conditioned on case01 output |
| case04 | `fold_from_sequence` | Boltz2 conf | Fold poly-glycine, verify confidence metrics |
| antibody | `fold_from_sequence` | Boltz2 conf | Fold trastuzumab VH (121 residues) |
| case05 | `fold_from_sequence` | Boltz2 aff | Fold poly-glycine, verify affinity metrics |
| case06 | `fold_from_structure` | Boltz2 aff | Re-fold case03 structure with affinity |

Each case validates:
- Geometry sanity (via `assert_geometry_sane_atom37!`)
- String output generation (PDB atom14, PDB atom37, mmCIF)
- File output writing
- Confidence/affinity metrics where applicable

### Expected Output

```
REPL API TEST SUMMARY (CPU)
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

## Prerequisites

- BoltzGen.jl and Onion.jl repos
- `runfromhere` Julia project environment with all dependencies
- Mixed SMILES case (case03) requires `MoleculeFlow` extension and initializes `.CondaPkg` in `runfromhere`
- Model weights are downloaded from HuggingFace on first use
- GPU tests require CUDA.jl and cuDNN.jl

## YAML Inputs

Committed in this repo:
- `examples/protein_binding_small_molecule/chorismite.yaml` (case02)
- `examples/mixed_protein_dna_smiles_smoke_v1.yaml` (case03)

## Legacy: Multi-Process Script Tests

The individual scripts can still be run standalone (see `scripts/` directory). Each script
requires its own Julia process and argument parsing. The REPL API tests above are preferred
as they are faster (single compilation) and test the programmatic API.

Individual script commands are documented in `CLAUDE.md` under "How to run working tests".
