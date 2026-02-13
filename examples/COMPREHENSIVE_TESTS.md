# Comprehensive Tests (HF-only)

This runbook executes the current Julia-only comprehensive matrix and geometry checks.

It is intentionally configured so that:
- Model weights are resolved from HuggingFace only (no local weight path loading).
- CCD molecule cache (`mols_cache.jld2`) is resolved from HuggingFace dataset only (no local cache path loading).

## Required repos and environment

Required local repos:
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl`
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/Onion.jl`

Required Julia project env:
- `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/runfromhere`

Notes:
- Mixed SMILES case requires `MoleculeFlow` extension/runtime and will initialize `.CondaPkg` in `runfromhere`.
- First run may download HF checkpoints and CCD cache; subsequent runs use HF cache.

## YAML inputs committed in this repo

- Protein + small-molecule: `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/examples/protein_binding_small_molecule/chorismite.yaml`
- Protein + DNA + SMILES: `/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/examples/mixed_protein_dna_smiles_smoke_v1.yaml`

## One-command comprehensive run

```bash
set -euo pipefail
OUT="/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/comprehensive_test_outputs/comprehensive_test_rerun_$(date +%Y%m%d_%H%M%S)_hf_singlecmd"
mkdir -p "$OUT"
JULIA_BIN="/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia"
PROJECT="/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/runfromhere"
DEPOT="/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/boltzgen_cache/julia_depot:/Users/benmurrell/.julia"
SCRIPT_DIR="/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/scripts"
GEOM="/Users/benmurrell/JuliaM3/BoltzGenJuliaPort/parity_testing_scripts/check_output_pdb_geometry.jl"

JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$SCRIPT_DIR/run_design_from_sequence.jl" --length 12 --steps 20 --recycles 2 --seed 7 --weights boltzgen1_diverse_state_dict.safetensors --out-pdb "$OUT/case01_design_protein_only_atom14.pdb" --out-pdb-atom37 "$OUT/case01_design_protein_only_atom37.pdb" --out-cif "$OUT/case01_design_protein_only_atom37.cif" > "$OUT/case01.main.log" 2>&1
JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$SCRIPT_DIR/run_design_from_yaml.jl" --yaml /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/examples/protein_binding_small_molecule/chorismite.yaml --steps 20 --recycles 2 --seed 7 --weights boltzgen1_diverse_state_dict.safetensors --out-pdb "$OUT/case02_design_protein_small_molecule_atom14.pdb" --out-pdb-atom37 "$OUT/case02_design_protein_small_molecule_atom37.pdb" --out-cif "$OUT/case02_design_protein_small_molecule_atom37.cif" > "$OUT/case02.main.log" 2>&1
JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$SCRIPT_DIR/run_design_from_yaml.jl" --yaml /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/BoltzGen.jl/examples/mixed_protein_dna_smiles_smoke_v1.yaml --steps 20 --recycles 2 --seed 7 --weights boltzgen1_diverse_state_dict.safetensors --out-pdb "$OUT/case03_design_protein_dna_smiles_atom14.pdb" --out-pdb-atom37 "$OUT/case03_design_protein_dna_smiles_atom37.pdb" --out-cif "$OUT/case03_design_protein_dna_smiles_atom37.cif" > "$OUT/case03.main.log" 2>&1
JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$SCRIPT_DIR/run_fold_from_sequence.jl" --sequence GGGGGGGGGGGGGG --steps 20 --recycles 2 --seed 7 --weights boltz2_conf_final_state_dict.safetensors --out-pdb "$OUT/case04_fold_sequence_confidence_atom14.pdb" --out-pdb-atom37 "$OUT/case04_fold_sequence_confidence_atom37.pdb" --out-cif "$OUT/case04_fold_sequence_confidence_atom37.cif" > "$OUT/case04.main.log" 2>&1
JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$SCRIPT_DIR/run_fold_from_sequence.jl" --sequence GGGGGGGGGGGGGG --steps 20 --recycles 2 --seed 7 --with-affinity true --weights boltz2_aff_state_dict.safetensors --out-heads "$OUT/case05_fold_sequence_affinity_heads.txt" --out-pdb "$OUT/case05_fold_sequence_affinity_atom14.pdb" --out-pdb-atom37 "$OUT/case05_fold_sequence_affinity_atom37.pdb" --out-cif "$OUT/case05_fold_sequence_affinity_atom37.cif" > "$OUT/case05.main.log" 2>&1
JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$SCRIPT_DIR/run_fold_from_structure.jl" --target "$OUT/case03_design_protein_dna_smiles_atom14.pdb" --steps 20 --recycles 2 --seed 7 --with-affinity true --weights boltz2_aff_state_dict.safetensors --out-heads "$OUT/case06_fold_structure_affinity_heads.txt" --out-pdb "$OUT/case06_fold_structure_affinity_atom14.pdb" --out-pdb-atom37 "$OUT/case06_fold_structure_affinity_atom37.pdb" --out-cif "$OUT/case06_fold_structure_affinity_atom37.cif" > "$OUT/case06.main.log" 2>&1
JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$SCRIPT_DIR/run_target_conditioned_design.jl" --target "$OUT/case01_design_protein_only_atom14.pdb" --design-length 8 --steps 20 --recycles 2 --seed 7 --weights boltzgen1_diverse_state_dict.safetensors --out-pdb "$OUT/case07_target_conditioned_design_atom14.pdb" --out-pdb-atom37 "$OUT/case07_target_conditioned_design_atom37.pdb" --out-cif "$OUT/case07_target_conditioned_design_atom37.cif" > "$OUT/case07.main.log" 2>&1

JULIA_DEPOT_PATH="$DEPOT" "$JULIA_BIN" --project="$PROJECT" "$GEOM" \
  "$OUT/case01_design_protein_only_atom14.pdb" "$OUT/case01_design_protein_only_atom37.pdb" \
  "$OUT/case02_design_protein_small_molecule_atom14.pdb" "$OUT/case02_design_protein_small_molecule_atom37.pdb" \
  "$OUT/case03_design_protein_dna_smiles_atom14.pdb" "$OUT/case03_design_protein_dna_smiles_atom37.pdb" \
  "$OUT/case04_fold_sequence_confidence_atom14.pdb" "$OUT/case04_fold_sequence_confidence_atom37.pdb" \
  "$OUT/case05_fold_sequence_affinity_atom14.pdb" "$OUT/case05_fold_sequence_affinity_atom37.pdb" \
  "$OUT/case06_fold_structure_affinity_atom14.pdb" "$OUT/case06_fold_structure_affinity_atom37.pdb" \
  "$OUT/case07_target_conditioned_design_atom14.pdb" "$OUT/case07_target_conditioned_design_atom37.pdb" \
  > "$OUT/geometry_all.log" 2>&1

echo "OUT_DIR=$OUT"
echo "GEOM_PASS=$(grep -c '^PASS' "$OUT/geometry_all.log" || true)"
echo "GEOM_FAIL=$(grep -c '^FAIL' "$OUT/geometry_all.log" || true)"
```

## Expected outputs

Per case:
- `caseNN.main.log`
- `*_atom14.pdb`
- `*_atom37.pdb`
- `*_atom37.cif`
- affinity cases also write `*_heads.txt`

Global:
- `geometry_all.log`

## Quick sanity checks

```bash
OUT=<your_output_dir>
rg -n "model loaded \(weights=" "$OUT"/case0{1,2,3,4,5,6,7}.main.log
rg -n "^PASS|^FAIL" "$OUT/geometry_all.log"
```

The `weights=` value should appear as HF cached artifact identifiers (hash-like names), not local absolute file paths.
