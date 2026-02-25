# Status & Known Issues

Last updated: 2026-02-13

## Summary

BoltzGen.jl is a Julia port of the BoltzGen/Boltz2 protein design and structure prediction pipeline. The codebase supports both CPU and GPU inference with a REPL-friendly API for interactive use. All 8 test cases pass on both CPU and GPU.

## What Works

### Design (BoltzGen1)
- De novo protein design (`design_from_sequence`, `denovo_sample`)
- Sequence redesign with per-residue design masks
- Cyclic peptide design
- YAML-driven mixed-complex design (protein + DNA/RNA + small molecules)
- Target-conditioned design from PDB/CIF structures

### Folding (Boltz2)
- Fold from sequence with confidence scoring (`fold_from_sequence`)
- Multi-chain complex folding (`fold_from_sequences`)
- Structure re-folding/refinement (`fold_from_structure`)
- Confidence metrics: pTM, ipTM, complex pLDDT, complex ipLDDT
- Binding affinity prediction (with `affinity=true`)

### Output & Validation
- PDB output (atom14 and atom37 representations)
- mmCIF output
- Geometry sanity checking
- Bond length validation

### GPU
- All functions run on GPU via `gpu=true` at model load time
- Results always returned on CPU
- No scalar indexing; all operations are vectorized

## Known Limitations

### 1. YAML/Parser Parity with Python (Incomplete)

Full Python parity for preprocessing and parser paths is not yet achieved. In a parity sweep of 73 YAML inputs:
- 16 pass strict parity (exact tensor match with Python)
- 50 have matching reference shapes but feature-level mismatches
- Remaining cases have shape mismatches or other issues

The main gaps are in nonpolymer/file-entity handling edge cases within the YAML parser and feature construction pipeline.

### 2. Fold-from-Structure Ligand Tokenization

When reloading generated PDB files containing small molecules, `load_structure_tokens` can collapse ligand tokenization:

- **Symptom**: `UNK`-labeled chain atoms become collapsed into a single token and emitted as `ATOM` instead of `HETATM`
- **Root cause**: `load_structure_tokens` groups records by `(chain, residue index, insertion code, comp_id)`, which merges atomized ligand tokens. Additionally, `comp_id == "UNK"` is mapped to `PROTEIN` before considering `is_het` flags.
- **Consequences**: Affinity auto-mask (nonpolymer-based) can be empty; ligand record type switches from `HETATM` to `ATOM`; geometry can degrade (near-overlapping atoms)

**Workaround**: Start from original YAML/SMILES or structure files that preserve ligand identity rather than re-loading generated PDB files.

### 3. Geometry Gate Threshold

The geometry sanity checker (`assert_geometry_sane_atom37!`) only fails when `min_nearest_neighbor < 0.01 A`. Near-overlaps (e.g., `0.0448 A`) currently pass. Ligand-specific HETATM checks are skipped if ligand atoms were emitted as ATOM records.

### 4. Confidence/Affinity on CPU

Confidence and affinity heads always run on CPU (data is moved from GPU before these forward passes). This is a deliberate design choice for numerical stability.

### 5. Model Family Constraints

- **Boltz2 is fold-only**: design masks and design chain appending are rejected. Use BoltzGen1 for generative design.
- **No automatic MSA search**: the package accepts pre-computed MSAs (via `msa_file` or `msa_sequences`) but does not run MSA search tools. You must provide alignments from external tools (e.g., MMseqs2, HHblits, Jackhmmer).

## Parity Rule

A case is considered **not at parity** if it does not exactly match Python reference features/tensors, even if it runs to completion without errors. "No runtime error" alone is not sufficient for parity.

## Fix Priorities

1. Preserve nonpolymer token granularity when loading structure files (prevent ligand token collapse on PDB reload)
2. Correct `UNK` + HETATM classification in structure parsing so nonpolymer identity is retained
3. Tighten geometry checks for near-overlaps and add checks independent of HETATM labeling
4. Continue YAML parser parity improvements toward full Python match
