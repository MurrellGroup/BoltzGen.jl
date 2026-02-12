# Developer Notes (Handoff)

Last updated: 2026-02-12

This document is the operational handoff for the BoltzGen Julia port workspace at:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort`

It covers:

- What each repo/folder is for.
- Which weight files exist and how they are used.
- How to run Julia end-to-end examples.
- How to run parity checks and geometry checks.
- How to set up official Python BoltzGen on another system.
- What is not fully implemented yet.

## 1) Workspace Layout

Primary folders:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache`

What they are:

- `BoltzGen.jl`: Julia port implementation and Julia run scripts.
- `Onion.jl`: Julia ML dependency used by BoltzGen.jl.
- `boltzgen`: Official Python reference implementation.
- `parity_testing_scripts`: Julia/Python parity export + comparison scripts.
- `boltzgen_cache`: Local weights, feature dumps, generated structures, parity reports.

Status docs already in-repo:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/current_status.md`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/feature_status.md`

## 2) Runtime Rules and Invocation Pattern

Use this Julia invocation pattern in this workspace:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl <absolute_script_path> <args...>
```

Operational constraints in this workspace:

- Run Julia commands one-by-one (no parallel Julia jobs here).
- Avoid batching many Julia launches in one shell loop.
- Avoid Julia `-e` checks unless necessary.
- Never use broad kill commands like `killall julia`.
- Keep strict behavior: no silent fallback logic in core code.

## 3) Weight Files

### 3.1 Official Python artifact names

The Python CLI maps these artifact keys in:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src/boltzgen/cli/boltzgen.py`

Model artifacts:

- `design-diverse` -> `huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt`
- `design-adherence` -> `huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt`
- `inverse-fold` -> `huggingface:boltzgen/boltzgen-1:boltzgen1_ifold.ckpt`
- `folding` -> `huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt`
- `affinity` -> `huggingface:boltzgen/boltzgen-1:boltz2_aff.ckpt`

Data artifact:

- `moldir` -> `huggingface:boltzgen/inference-data:mols.zip`

### 3.2 Julia safetensors in local cache

Current files under `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache`:

- `boltzgen1_diverse_state_dict.safetensors`
- `boltzgen1_adherence_state_dict.safetensors`
- `boltz2_conf_final_state_dict.safetensors`
- `boltz2_aff_state_dict.safetensors`
- `boltz2_conf_merged_with_diverse.safetensors`
- `boltz2_aff_state_dict_module1remap.safetensors`
- raw `.ckpt` files may also exist (`boltz2_conf_final.ckpt`, `boltz2_aff.ckpt`)

How these are used today:

- `boltzgen1_diverse_state_dict.safetensors`
  - Default for design scripts.
  - Supports structure diffusion/design pathways.
- `boltzgen1_adherence_state_dict.safetensors`
  - Alternative design checkpoint.
- `boltz2_conf_final_state_dict.safetensors`
  - Default for fold scripts (with confidence path).
- `boltz2_aff_state_dict.safetensors`
  - Fold + affinity-enabled runs.
- `boltz2_conf_merged_with_diverse.safetensors`
  - Parity/experimental merged checkpoint used in some parity checks.
- `boltz2_aff_state_dict_module1remap.safetensors`
  - Experimental/remapped variant for affinity-key experiments.

### 3.3 Current Julia weight resolver behavior

Implemented in:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/src/checkpoint.jl`

Behavior:

- Default repo: `MurrellLab/BoltzGen.jl` (regular HuggingFace model repo).
- `resolve_weights_path(...)` accepts:
  - explicit local path (used directly if file exists), or
  - safetensors filename / alias (downloaded via `HuggingFaceApi.hf_hub_download` and cached).
- Aliases:
  - `boltzgen1_diverse`
  - `boltzgen1_adherence`
  - `boltz2_conf_final`
  - `boltz2_aff`

Defaults by family:

- `boltzgen1` -> `boltzgen1_diverse_state_dict.safetensors`
- `boltz2` with affinity false -> `boltz2_conf_final_state_dict.safetensors`
- `boltz2` with affinity true -> `boltz2_aff_state_dict.safetensors`

## 4) Small-Molecule / CCD Data

### 4.1 Raw molecule store

Path:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols`

Contents:

- ~45k `.pkl` files (`45227` currently), one per code.
- Each file is an RDKit `Mol` pickle in Python format.

### 4.2 Julia-native cache

Path:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols_cache.jld2`

Runtime loader:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/src/ccd_cache.jl`

Runtime behavior:

- Default JLD2 cache path is `.../boltzgen_cache/mols_cache.jld2`.
- Override via env var `BOLTZGEN_MOLS_JLD2`.
- Hard errors if cache missing or malformed.

### 4.3 Build/update JLD2 cache (one-time/offline preprocessing)

Python export step:

```bash
python3 /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/export_mols_pkl_to_jsonl.py \
  --moldir /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols \
  --out /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols_export.jsonl
```

Julia build step:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/build_mols_jld2_cache.jl \
  --jsonl /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols_export.jsonl \
  --out /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols_cache.jld2
```

### 4.4 Source and license notes for molecule data

Source in official Python CLI is the HuggingFace dataset artifact:

- `huggingface:boltzgen/inference-data:mols.zip`

Important licensing note:

- The local codebase does not provide a single explicit license statement for `mols.zip` payload itself.
- Do not assume redistribution terms for molecule data from code license alone.
- Verify the upstream HuggingFace repository card/license terms for `boltzgen/inference-data` before redistribution.

## 5) Julia End-to-End Run Scripts

Scripts in:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts`

### 5.1 Design from sequence

Script:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_design_from_sequence.jl`

Example:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_design_from_sequence.jl \
  --length 30 --steps 60 --recycles 2 --seed 7 \
  --out-pdb /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_seq_atom14.pdb \
  --out-pdb-atom37 /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_seq_atom37.pdb \
  --out-cif /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_seq_atom37.cif
```

### 5.2 Design from YAML (includes mixed biomolecule cases)

Script:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_design_from_yaml.jl`

Example:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_design_from_yaml.jl \
  --yaml /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/examples/mixed_protein_dna_smiles_smoke_v1.yaml \
  --steps 60 --recycles 2 --seed 7 \
  --out-pdb /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_yaml_atom14.pdb \
  --out-pdb-atom37 /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_yaml_atom37.pdb \
  --out-cif /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_yaml_atom37.cif
```

### 5.3 Fold from sequence

Script:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_fold_from_sequence.jl`

Example:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_fold_from_sequence.jl \
  --sequence GGGGGGGGGG --steps 60 --recycles 2 --seed 7 \
  --out-pdb /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_fold_seq_atom14.pdb \
  --out-pdb-atom37 /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_fold_seq_atom37.pdb \
  --out-cif /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_fold_seq_atom37.cif
```

### 5.4 Fold from structure

Script:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_fold_from_structure.jl`

Example:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_fold_from_structure.jl \
  --target /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example/vanilla_protein/1g13.cif \
  --steps 60 --recycles 2 --seed 7 \
  --out-pdb /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_fold_struct_atom14.pdb \
  --out-pdb-atom37 /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_fold_struct_atom37.pdb \
  --out-cif /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_fold_struct_atom37.cif
```

### 5.5 Target-conditioned design

Script:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_target_conditioned_design.jl`

Example:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_target_conditioned_design.jl \
  --target /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example/vanilla_protein/1g13.cif \
  --design-length 20 --steps 60 --recycles 2 --seed 7 \
  --out-pdb /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_target_cond_atom14.pdb \
  --out-pdb-atom37 /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_target_cond_atom37.pdb \
  --out-cif /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_target_cond_atom37.cif
```

### 5.6 Required mixed-complex smoke test

Script:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_mixed_protein_dna_smiles_smoke.jl`

Command:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts/run_mixed_protein_dna_smiles_smoke.jl \
  --steps 60 --recycles 2 --seed 7 --tag dev
```

This runs generation plus geometry checks for a protein + DNA + SMILES ligand case.

## 6) Geometry Checks

Primary scripts:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_output_pdb_geometry.jl`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_dna_geometry.jl`

Commands:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_output_pdb_geometry.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_yaml_atom37.pdb
```

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_dna_geometry.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/dev_design_yaml_atom37.pdb
```

Current caveat:

- Geometry gates catch major failures but are not perfect. A near-overlap can still pass if above hard threshold.
- See `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/feature_status.md` for the known fold+ligand collapse case and checker limitations.

## 7) Parity Philosophy and Acceptance Rule

Hard rule in this workspace:

- A run that "did not crash" is not success.
- "Fail" means "did not exactly match Python features/tensors".

Canonical status reference:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/current_status.md`

## 7.1 Existing Comprehensive Regression Artifacts

Most recent tracked comprehensive report paths:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_report_comprehensive_test_v2.txt`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_report_comprehensive_test.txt`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_report_comprehensive_test_direct.txt`

Most recent output directory:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_outputs/comprehensive_test_v2`

Cases present in that matrix (per current feature status):

- `case01_design_protein_only`
- `case02_design_protein_small_molecule`
- `case03_design_protein_dna_smiles`
- `case04_fold_sequence_confidence`
- `case05_fold_structure_protein_only`
- `case06_fold_structure_affinity`
- `case07_target_conditioned_design`

Note:

- Case names differ slightly between some historical report versions; treat the report text itself as source-of-truth for that run.

## 8) Parity Script Inventory

Located in:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts`

Main comparison scripts:

- `check_input_parity.jl`
  - Compares feature tensors from NPZ against Julia-loaded equivalents.
- `check_layer_parity.jl`
  - Layer-level fixture parity (attention/triangle/transition/etc).
- `check_boltzgen_model_parity.jl`
  - Lite-model fixture parity.
- `check_boltzgen_full_parity.jl`
  - Full-model fixture parity.
- `check_design_feature_parity.jl`
  - Design feature parity from Python-exported YAML NPZ.
- `check_yaml_parser_parity.jl`
  - Single-YAML parser/feature parity, supports replay plans.
- `check_yaml_parser_parity_sweep.jl`
  - Full YAML sweep parity over the `example/` tree with strict pass/fail.
- `check_diffusion_input_parity.jl`
  - Trunk/diffusion-input internal tensor parity.
- `check_confidence_affinity_parity.jl`
  - Confidence + affinity head output parity (with matching dumps/checkpoints).
- `check_output_pdb_geometry.jl`
  - General geometry sanity checks.
- `check_dna_geometry.jl`
  - DNA/RNA backbone-specific geometry checks.

Python export helpers:

- `export_yaml_features.py`
- `export_yaml_features_sweep.py`
- `export_diffusion_inputs.py`
- `export_boltzgen_model.py`
- `export_boltzgen_full.py`
- `export_ckpt_state_dict.py`

Shared Julia helper:

- `run_full_inference.jl`

## 9) Recommended Parity Workflow (Current)

### 9.1 Single YAML debug path

Python export + plans:

```bash
PYTHONPATH=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/.venv/bin/python \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/export_yaml_features.py \
  --yaml /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example/vanilla_protein/1g13prot.yaml \
  --moldir /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols \
  --out /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.npz \
  --plan-out /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.sampling_plan.yaml \
  --conformer-plan-out /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.conformer_plan.yaml
```

Julia check:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_yaml_parser_parity.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example/vanilla_protein/1g13prot.yaml \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.npz \
  1 \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.sampling_plan.yaml \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_feats_test_1g13prot_plan_v1.conformer_plan.yaml
```

### 9.2 Full YAML sweep

Python sweep export:

```bash
PYTHONPATH=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/.venv/bin/python \
/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/export_yaml_features_sweep.py \
  --yaml-root /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example \
  --moldir /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/mols \
  --outdir /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_current
```

Julia sweep check:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_yaml_parser_parity_sweep.jl \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/example \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_current \
  1 \
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/yaml_parser_parity_sweep_current.tsv
```

### 9.3 Downstream parity order after parser parity is clean

Run in this order:

- `check_design_feature_parity.jl`
- `check_diffusion_input_parity.jl`
- `check_confidence_affinity_parity.jl`
- `check_boltzgen_model_parity.jl`
- `check_boltzgen_full_parity.jl`
- geometry checks on generated structures

### 9.4 Parity Command Cookbook (Concrete Examples)

Design feature parity:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_design_feature_parity.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_yaml_feats_sweep_current/py_feats_vanilla_protein__1g13prot.npz
```

Diffusion-input parity:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_diffusion_input_parity.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_diffusion_inputs_mini10_yaml_fresh_s100_r3_seed7_v1.npz \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/generated_from_yaml_minimal_v1_features.npz \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/boltzgen1_diverse_state_dict.safetensors
```

Confidence/affinity head parity:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_confidence_affinity_parity.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/py_heads_dump_mini10_aff_conf_v1.npz \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/boltz2_conf_merged_with_diverse.safetensors
```

Fixture model parity checks:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_boltzgen_model_parity.jl
```

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_boltzgen_full_parity.jl
```

Input parity:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_input_parity.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/features_1g13prot.npz
```

Layer parity:

```bash
julia --project=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl \\
  /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_layer_parity.jl
```

## 10) Setting Up Official Python BoltzGen on a Separate System

Reference README:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/README.md`

### 10.1 Basic install

Requirements:

- Python `>=3.11`.
- GPU recommended for actual design workloads.

Minimal:

```bash
pip install boltzgen
```

From source:

```bash
git clone https://github.com/HannesStark/boltzgen
cd boltzgen
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 10.2 Download official artifacts

```bash
boltzgen download all
```

This fetches checkpoints and `moldir` artifact (molecule canonical library).

### 10.3 Run official Python CLI

Sanity-check a design spec:

```bash
boltzgen check /path/to/design.yaml
```

Run pipeline:

```bash
boltzgen run /path/to/design.yaml --output /path/to/output --protocol protein-anything
```

### 10.4 Parity setup (Python side for this workspace)

For this specific workspace’s parity scripts, use:

- Python env at `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/.venv`
- `PYTHONPATH=/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen/src`

Then run the export scripts in `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts`.

## 11) Current Known Gaps / Not Fully Implemented

From current status docs and latest tracked runs:

- YAML/parser parity is still incomplete.
- Full strict parity across all official example YAMLs is not achieved.
- Nonpolymer/file-entity handling still has mismatch edge cases.
- Fold-from-structure can collapse ligand tokenization in known scenarios.
- Geometry checks are useful but not perfect as strict physical validators.
- Boltz2 is intentionally fold-only in current Julia run-script guards.
- Comprehensive test matrix outputs exist in cache reports, but there is no fully productized committed "single-command" comprehensive test driver script yet.

Track details in:

- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/current_status.md`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/feature_status.md`

## 12) Practical Troubleshooting

- If a script appears stuck early, verify Julia launcher/Conda lock behavior and rerun that single case (not parallel runs).
- Confirm exact checkpoint used in logs and parity dump metadata before comparing heads.
- For molecule issues, verify `mols_cache.jld2` exists and that `BOLTZGEN_MOLS_JLD2` is set correctly if overriding default path.
- Prefer absolute paths everywhere in this workspace.
