# Current Status: BoltzGen.jl

Last updated: 2026-02-13

## Summary

BoltzGen.jl is a Julia port of the BoltzGen/Boltz2 protein design and structure prediction pipeline.
The codebase supports both CPU and GPU inference, with a REPL-friendly API for interactive use.

## What Works

### REPL API (all 8 test cases pass on CPU and GPU)

- **Design functions** (BoltzGen1 model):
  - `design_from_sequence` — de novo protein design and sequence redesign
  - `design_from_yaml` — design from YAML specification (protein, DNA, RNA, SMILES, mixed)
  - `target_conditioned_design` — design conditioned on target structure
  - `denovo_sample` — simplest de novo sampling path

- **Fold functions** (Boltz2 model):
  - `fold_from_sequence` — structure prediction from amino acid sequence
  - `fold_from_structure` — re-fold/refine from PDB/CIF file

- **Output functions**:
  - `output_to_pdb`, `output_to_pdb_atom37`, `output_to_mmcif` — string output
  - `write_outputs` — file output (PDB atom14, PDB atom37, mmCIF)
  - `confidence_metrics` — extract pTM, ipTM, pLDDT, affinity scores

- **GPU inference**:
  - All functions run on GPU via `gpu=true` flag at model load time
  - Results are always returned on CPU
  - No scalar indexing; all operations are vectorized or use batched_mul

### End-to-end test script

Single-session test covering all 8 cases:
```bash
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl        # CPU
julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu  # GPU
```

### Legacy scripts (still functional)

Individual scripts in `BoltzGen.jl/scripts/` continue to work for standalone use.

## Known Limitations

### Parser/feature parity with Python

Full Python parity for preprocessing/parser paths is not yet achieved, particularly for:
- NONPOLYMER/file-entity handling edge cases
- Fold-from-structure can collapse ligand tokenization when reloading PDB files
- Some YAML specification features have edge-case mismatches

### Confidence/affinity module device

Confidence and affinity heads run on CPU (moved from GPU before forward pass).
This is a deliberate design choice for numerical stability and simplicity.

## Documentation

- REPL API reference: `examples/REPL_API.md`
- Comprehensive tests: `examples/COMPREHENSIVE_TESTS.md`
- Feature-level status: `feature_status.md`
