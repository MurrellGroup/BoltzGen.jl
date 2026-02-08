# BoltzGen.jl Feature Status (Julia vs Python)

As of **February 8, 2026**.

## Parity Rule (Hard Requirement)
- A case is considered **failed** if it does not exactly match Python features/tensors, even if it runs to completion.
- "No runtime error" is not success.

## Current High-Level Status
- End-to-end Julia runs produce outputs across design/fold modes (see comprehensive test below).
- Full Python parity is **not** yet achieved for preprocessing/parser paths, especially nonpolymer/file-entity handling.

## Comprehensive Test (Julia-only smoke/regression)
- Report: `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_report_comprehensive_test_v2.txt`
- Output dir: `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_outputs/comprehensive_test_v2`
- Cases currently included:
  - `case01_design_protein_only`
  - `case02_design_protein_small_molecule`
  - `case03_design_protein_dna_smiles`
  - `case04_fold_sequence_confidence`
  - `case05_fold_structure_protein_only`
  - `case06_fold_structure_affinity`
  - `case07_target_conditioned_design`
- Completion status in report: `7/7` completed and wrote PDB/mmCIF outputs.

## Known Working Capabilities (Execution-Level)
- `run_design_from_sequence.jl` works for protein design with BoltzGen1 design checkpoints.
- `run_design_from_yaml.jl` works for mixed protein/DNA/SMILES example inputs.
- `run_target_conditioned_design.jl` works for target-conditioned design in BoltzGen1 mode.
- `run_fold_from_sequence.jl` and `run_fold_from_structure.jl` run in Boltz2 fold mode.
- Confidence and affinity heads can run when checkpoint and masks are configured correctly.

## Known Gaps / Non-Parity Areas

### 1) YAML/parser parity still incomplete
- Latest tracked sweep file:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/yaml_parser_parity_sweep_gpt53_unresolved_ccd_entityfix14.tsv`
- Summary from that file:
  - rows (YAMLs): `73`
  - strict `ok=true`: `16`
  - `ref_shape_ok=true`: `50`
- Interpretation: parser/preprocessing remains a blocking parity gap.

### 2) Fold-from-structure with small molecule can degrade ligand representation (case06)
- Specific affected output:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_outputs/comprehensive_test_v2/case06_fold_structure_affinity_atom14.pdb`
- Observed symptom:
  - `UNK` chain `C` atoms become tightly collapsed (`min_nn ~ 0.0448 A`) and are emitted as `ATOM` instead of `HETATM`.
- Diagnostic proof (Julia-only):
  - YAML parse of `mixed_protein_dna_smiles_smoke_v1.yaml`:
    - total tokens `33`
    - chain `C` tokens `13`
    - all chain `C` tokens are `NONPOLYMER`, each with one atom
  - Reloading `case03_design_protein_dna_smiles_atom14.pdb` with `load_structure_tokens(...; include_nonpolymer=true)`:
    - total tokens `21`
    - chain `C` tokens `1`
    - chain `C` becomes `PROTEIN` `UNK` with `13` atoms on one token
- Root cause (current code behavior):
  - `load_structure_tokens` groups records by `(chain, residue index, insertion code, comp_id)`, which collapses atomized ligand tokens into a single token on PDB reload.
  - `_token_and_mol_type_from_comp_id` maps `comp_id == "UNK"` to `PROTEIN` before considering `is_het`/`include_nonpolymer`.
- Consequences:
  - Affinity auto-mask (nonpolymer-based) can be empty.
  - Ligand record type can switch from `HETATM` to `ATOM`.
  - Geometry can degrade because the model sees a different tokenization than intended.

### 3) Geometry gate is too permissive for near-overlap failures
- Current checker:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_output_pdb_geometry.jl`
- It only fails when `min_nn < 0.01`, so `0.0448 A` still passes.
- Ligand-specific HETATM checks are skipped if ligand atoms were emitted as `ATOM`.

### 4) Model-family limitations (intentional but important)
- Boltz2 in these scripts is fold-only; design masks/appended design chains are rejected.
- This is expected behavior in current script guards, but should remain explicit in docs.

## Practical Guidance (Until Fixed)
- Do not treat generated atom14 PDBs with `UNK` ligands as lossless fold+affinity re-entry inputs.
- Prefer starting from original YAML/SMILES or structure files that preserve ligand identity/mapping.
- Keep using strict parity scripts as the acceptance gate before claiming correctness.

## Next Fix Priorities
1. Preserve nonpolymer token granularity when loading structure files (no ligand token collapse on reload).
2. Correct `UNK` + HETATM classification in structure parsing so nonpolymer identity is not dropped.
3. Tighten geometry checks for near-overlaps and add checks independent of `HETATM` labeling.
4. Re-run full YAML parity sweep and update this document with new counts.
