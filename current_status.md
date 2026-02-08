# Current Status: BoltzGen Julia Parser/Input Parity

## Parity Semantics (Do Not Relax)
- In this project, **"fail" means "did not exactly match Python features"**, even if execution completed without runtime errors.
- A run that does not crash but has any feature mismatch is still a hard failure.
- Success means exact parity on compared feature tensors/arrays, not merely "no exception".

## Purpose
Port BoltzGen inference stack to Julia with parity against Python, including YAML-driven design/folding inputs. Current focus is parser/input parity (before full-model parity can be trusted).

## Companion Status Doc
- Feature-level working/incomplete matrix (including fold+small-molecule round-trip issue):
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/feature_status.md`

## Repos and Code Locations
- Julia implementation repo (committable):
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl`
- Python reference repo:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen`
- Parity scripts (currently in workspace root, not inside `BoltzGen.jl/.git`):
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts`

### Python reference files for this problem
- Main YAML parser logic:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src/boltzgen/data/parse/schema.py`
- YAML prediction data path calling parser + featurizer:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src/boltzgen/task/predict/data_from_yaml.py`
- Feature construction path used downstream:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src/boltzgen/data/feature/featurizer.py`

### Julia files touched for replay and parser parity
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/src/yaml_parser.jl`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/src/features.jl`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_design_from_yaml.jl`

## Replay Functionality (Important)
Replay is **NOT enabled by default**.

- Default parser behavior:
  - `BoltzGen.parse_design_yaml(...; sampling_plan=nothing)`
  - This is normal inference behavior.
- Replay-enabled behavior (parity/dev only):
  - Pass a sampling plan generated from Python.
  - Parser hard-errors on mismatch/exhaustion/unused decisions (no fallback).

### How replay is enabled
- Single check:
  - Provide 4th argument to `check_yaml_parser_parity.jl` as sampling plan YAML.
- Sweep check:
  - `check_yaml_parser_parity_sweep.jl` expects sidecars named `*.sampling_plan.yaml` beside each Python NPZ.

## Current State of the Problem
Replay reduced randomness-induced divergence significantly, but parser/file-entity parity is still not solved.

### Latest replay sweep outputs
- Python export + sampling plans:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_gpt53_v3`
- Julia replay parity report:
  - `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/yaml_parser_parity_sweep_gpt53_v3b.tsv`

### Headline metrics (v3b)
- Official YAMLs: 73
- Python-exportable YAMLs: 53
- Python-native failures: 20 (same as before; mostly `KeyError: 'entities'`, one missing CIF, one `no subchain E`)
- Compared YAMLs: 53
- `ref_shape_ok=true`: 28/53 (was 6/53 before replay)
- Remaining hard errors in Julia side for compared set: 8
  - NONPOLYMER empty atom-name override
  - NONPOLYMER frame recompute assumptions
  - one sampling-plan exhaustion case

## What Is Still Wrong (Most Likely)
1. File-entity mmCIF/PDB handling is still not Python-equivalent.
- Python `parse_file` does unresolved-residue trimming on included chain segments.
- Julia file-entity path appears to keep residues Python drops, causing persistent shape and mapping mismatches.

2. NONPOLYMER token/atom mapping edge cases are still mismatched.
- Errors like `NONPOLYMER token ... has empty atom-name override` indicate tokenization/build logic diverges before feature parity checks.

3. Frame recompute assumptions differ for some nonpolymer cases.
- `NONPOLYMER frame recompute requires one atom per token` appears in cases where Python carries atoms differently through parsing/tokenization.

4. `ref_pos` invariants remain high in many cases even with replay.
- Means structural/random-augmentation groups or atom-to-token mapping still diverge downstream of replay decisions.

5. One replay exhaustion remains (`design_spec_showcasing_all_functionalities.yaml`).
- Implies additional parser-path divergence in random-call sequencing, not just random values.

## Next Work (Priority Order)
1. Implement Python-equivalent unresolved residue trimming in Julia file-entity parse path.
2. Fix NONPOLYMER atom-name/atom-assignment parity in file + smiles mixed cases.
3. Align nonpolymer frame construction/recompute logic with Python behavior exactly.
4. Resolve replay decision sequencing mismatch for `design_spec_showcasing_all_functionalities.yaml`.
5. Re-run full replay sweep until non-random keys pass for all Python-exportable YAMLs.
6. Only then rerun downstream full-model parity and geometry parity checks.

## Exact Commands: YAML Parser/Input Parity

### Environment baseline
Use these exact runtimes:
- Python: `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/.venv/bin/python`
- Julia: `julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl`

### A) Single YAML: Python export + sampling plan
```bash
PYTHONPATH=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/.venv/bin/python \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/export_yaml_features.py \
  --yaml /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example/vanilla_protein/1g13prot.yaml \
  --moldir /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols \
  --out /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.npz \
  --plan-out /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.sampling_plan.yaml
```

### B) Single YAML: Julia parity check with replay enabled
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_yaml_parser_parity.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example/vanilla_protein/1g13prot.yaml \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.npz \
  1 \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.sampling_plan.yaml
```

### C) Full 73 YAML sweep: Python exports + plans
```bash
PYTHONPATH=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/.venv/bin/python \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/export_yaml_features_sweep.py \
  --yaml-root /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example \
  --moldir /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols \
  --outdir /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_gpt53_v3
```

### D) Full 73 YAML sweep: Julia replay parity
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_yaml_parser_parity_sweep.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_gpt53_v3 \
  1 \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/yaml_parser_parity_sweep_gpt53_v3b.tsv
```

## Downstream Parity Scripts to Re-run After Fixing YAML/Input Parity
Run in this order once YAML replay sweep is clean for all Python-exportable YAMLs.

1. YAML parser/input parity (gate)
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_yaml_parser_parity_sweep.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_gpt53_v3 \
  1 \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/yaml_parser_parity_sweep_gpt53_v3b.tsv
```

2. Design feature parity (single exported NPZ)
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_design_feature_parity.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_gpt53_v3/py_feats_vanilla_protein__1g13prot.npz
```

3. Diffusion-input parity
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_diffusion_input_parity.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_diffusion_inputs_mini10_yaml_fresh_s100_r3_seed7_v1.npz \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/generated_from_yaml_minimal_v1_features.npz \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/boltzgen1_diverse_state_dict.safetensors
```

4. Confidence/Affinity head parity
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_confidence_affinity_parity.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_heads_dump_mini10_aff_conf_v1.npz \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/boltz2_conf_merged_with_diverse.safetensors
```

5. Fixed-fixture model parity checks
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_boltzgen_model_parity.jl
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_boltzgen_full_parity.jl
```

6. Geometry checks on generated outputs
```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_output_pdb_geometry.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mixed_protein_dna_smiles_s60_r2_seed7_gpt53_fix_v2_atom37.pdb

julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_dna_geometry.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mixed_protein_dna_smiles_s60_r2_seed7_gpt53_fix_v2_atom37.pdb
```

## Notes for Next Engineer
- Do not enable replay for normal inference routes.
- Replay mode is only a dev/parity mechanism to force equivalent random branching.
- Keep strict error behavior; no parser fallbacks on unexpected conditions.
