#!/usr/bin/env julia
#
# Single-session REPL API test covering all 7 original cases + antibody fold.
# Models are loaded once and reused, avoiding per-script recompilation.
#
# Usage:
#   julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl [output_dir]
#   julia --project=runfromhere BoltzGen.jl/scripts/run_repl_api_tests.jl --gpu [output_dir]
#

using Onion
using Random
using MoleculeFlow  # Must be loaded before BoltzGen to trigger extension

# Parse --gpu flag
const USE_GPU = "--gpu" in ARGS
non_flag_args = filter(a -> a != "--gpu", ARGS)

if USE_GPU
    println("GPU mode enabled — loading CUDA and cuDNN...")
    using CUDA
    using cuDNN
    using OnionTile  # Activate cuTile-accelerated kernels (flash attention, layernorm, etc.)
    println("CUDA functional: ", CUDA.functional())
    CUDA.functional() || error("CUDA is not functional. Cannot run GPU tests.")
    println("GPU device: ", CUDA.device())
end

using BoltzGen

Onion.bg_set_training!(false)

const OUT = if length(non_flag_args) >= 1
    non_flag_args[1]
else
    joinpath(WORKSPACE_ROOT, USE_GPU ? "repl_api_gpu_test_outputs" : "repl_api_test_outputs")
end
mkpath(OUT)

const LOG_FILE = joinpath(OUT, "test_log.txt")
const LOG_IO = open(LOG_FILE, "w")

function tee(args...)
    println(stdout, args...)
    println(LOG_IO, args...)
end

function tee_flush()
    flush(LOG_IO)
end

const YAML_DIR = normpath(joinpath(@__DIR__, "..", "examples"))

# ── Test runner bookkeeping ─────────────────────────────────────────────────────

const results = Tuple{String,Symbol,String,Float64}[]  # (name, :PASS/:FAIL, detail, elapsed_s)

function run_case(f::Function, name::String)
    tee("\n", "="^60)
    tee("  CASE: ", name)
    tee("="^60)
    try
        elapsed = @elapsed f()
        push!(results, (name, :PASS, "", elapsed))
        tee("[PASS] ", name, " ($(round(elapsed; digits=2))s)")
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        push!(results, (name, :FAIL, msg, 0.0))
        tee("[FAIL] ", name, ": ", msg)
    end
    tee_flush()
end

function check_geometry(result::Dict; label::String="")
    stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"]; batch=1)
    tee("  geometry: n_atoms=", stats.n_atoms,
        " max_abs=", round(stats.max_abs; digits=2),
        " min_nn=", round(stats.min_nearest_neighbor; digits=3),
        " max_nn=", round(stats.max_nearest_neighbor; digits=3),
        " frac_nn<0.5=", round(stats.frac_nearest_lt_0p5; digits=3),
        " frac_ge900=", round(stats.frac_abs_ge900; digits=3),
        isempty(label) ? "" : "  ($label)")
    return stats
end

function check_bonds(result::Dict)
    BoltzGen.print_bond_length_report(result; io=stdout)
    BoltzGen.print_bond_length_report(result; io=LOG_IO)
end

function check_string_outputs(result::Dict)
    pdb14 = BoltzGen.output_to_pdb(result)
    pdb37 = BoltzGen.output_to_pdb_atom37(result)
    cif = BoltzGen.output_to_mmcif(result)
    length(pdb14) > 10 || error("output_to_pdb returned empty/tiny string ($(length(pdb14)) bytes)")
    length(pdb37) > 10 || error("output_to_pdb_atom37 returned empty/tiny string ($(length(pdb37)) bytes)")
    length(cif) > 10 || error("output_to_mmcif returned empty/tiny string ($(length(cif)) bytes)")
    tee("  string outputs: pdb14=$(length(pdb14))B, pdb37=$(length(pdb37))B, cif=$(length(cif))B")
end

# ── Phase 1: BoltzGen1 (design model) ──────────────────────────────────────────

tee("\n", "#"^60)
tee("  Loading BoltzGen1 design model", USE_GPU ? " (GPU)" : "", "...")
tee("#"^60)
gen = BoltzGen.load_boltzgen(; gpu=USE_GPU)
tee("Loaded: ", gen)

# Warmup: compile all code paths before timing
tee("\n  Warmup: compiling design code paths...")
warmup_t = @elapsed begin
    BoltzGen.design_from_sequence(gen, ""; length=5, steps=2, recycles=1, seed=1)
end
tee("  warmup completed in $(round(warmup_t; digits=1))s")
if USE_GPU; CUDA.synchronize(); end

# Case 01: Design protein only (de novo, length=12)
run_case("case01_design_protein_only") do
    result = BoltzGen.design_from_sequence(gen, ""; length=12, steps=200, recycles=2, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case01_design_protein_only"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
end

# Case 02: Design protein + small molecule (chorismite.yaml)
run_case("case02_design_protein_small_molecule") do
    yaml = joinpath(YAML_DIR, "protein_binding_small_molecule", "chorismite.yaml")
    result = BoltzGen.design_from_yaml(gen, yaml; steps=200, recycles=2, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case02_design_protein_small_molecule"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
end

# Case 03: Design protein + DNA + SMILES (mixed yaml)
run_case("case03_design_protein_dna_smiles") do
    yaml = joinpath(YAML_DIR, "mixed_protein_dna_smiles_smoke_v1.yaml")
    result = BoltzGen.design_from_yaml(gen, yaml; steps=200, recycles=2, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case03_design_protein_dna_smiles"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
end

# Case 07: Target conditioned design (uses case01 output)
run_case("case07_target_conditioned_design") do
    target_pdb = joinpath(OUT, "case01_design_protein_only_atom14.pdb")
    isfile(target_pdb) || error("case01 PDB not found at $target_pdb (did case01 fail?)")
    result = BoltzGen.target_conditioned_design(gen, target_pdb; design_length=8, steps=200, recycles=2, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case07_target_conditioned_design"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
end

# Release boltzgen1 model
tee("\nReleasing BoltzGen1 model...")
gen = nothing
GC.gc()
if USE_GPU
    CUDA.reclaim()
end

# ── Phase 2: Boltz2 confidence model ───────────────────────────────────────────

tee("\n", "#"^60)
tee("  Loading Boltz2 confidence model", USE_GPU ? " (GPU)" : "", "...")
tee("#"^60)
fold = BoltzGen.load_boltz2(; affinity=false, gpu=USE_GPU)
tee("Loaded: ", fold)

# Warmup: compile fold code paths before timing
tee("\n  Warmup: compiling fold code paths...")
warmup_t = @elapsed begin
    BoltzGen.fold_from_sequence(fold, "GGGGG"; steps=2, recycles=1, seed=1)
end
tee("  warmup completed in $(round(warmup_t; digits=1))s")
if USE_GPU; CUDA.synchronize(); end

# Case 04: Fold sequence with confidence (GGGGGGGGGGGGGG)
run_case("case04_fold_sequence_confidence") do
    result = BoltzGen.fold_from_sequence(fold, "GGGGGGGGGGGGGG"; steps=200, recycles=3, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case04_fold_sequence_confidence"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
    # Verify confidence metrics
    metrics = BoltzGen.confidence_metrics(result)
    tee("  confidence metrics: ", pairs(metrics))
    hasproperty(metrics, :ptm) || error("Missing ptm in confidence metrics")
    hasproperty(metrics, :iptm) || error("Missing iptm in confidence metrics")
end

# Case antibody: Fold trastuzumab VH with confidence
run_case("case_antibody_fold") do
    # Trastuzumab VH (Herceptin) — 121 residues
    vh_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    result = BoltzGen.fold_from_sequence(fold, vh_seq; steps=200, recycles=3, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case_antibody_fold"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
    metrics = BoltzGen.confidence_metrics(result)
    tee("  confidence metrics: ", pairs(metrics))
end

# Case MSA: Fold lysozyme (HEWL, 129 residues) with 100-sequence real MSA
run_case("case_msa_lysozyme_fold") do
    # Hen egg-white lysozyme mature sequence (UniProt P00698, 129 residues)
    hewl_seq = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    msa_file = joinpath(YAML_DIR, "lysozyme_msa.fasta")
    isfile(msa_file) || error("MSA file not found at $msa_file")

    # Also run without MSA for timing comparison
    tee("  Folding WITHOUT MSA...")
    t_no_msa = @elapsed begin
        result_no_msa = BoltzGen.fold_from_sequence(fold, hewl_seq; steps=200, recycles=3, seed=7)
    end
    if USE_GPU; CUDA.synchronize(); end
    BoltzGen.write_outputs(result_no_msa, joinpath(OUT, "case_msa_lysozyme_no_msa"))
    check_geometry(result_no_msa; label="no_msa")
    check_bonds(result_no_msa)
    metrics_no = BoltzGen.confidence_metrics(result_no_msa)
    tee("  no-MSA: pTM=$(round(metrics_no.ptm[1]; digits=3)), time=$(round(t_no_msa; digits=2))s")

    tee("  Folding WITH MSA (100 sequences)...")
    t_msa = @elapsed begin
        result_msa = BoltzGen.fold_from_sequence(fold, hewl_seq;
            steps=200, recycles=3, seed=7, msa_file=msa_file, max_msa_rows=100)
    end
    if USE_GPU; CUDA.synchronize(); end
    BoltzGen.write_outputs(result_msa, joinpath(OUT, "case_msa_lysozyme_with_msa"))
    check_geometry(result_msa; label="with_msa")
    check_bonds(result_msa)
    metrics_msa = BoltzGen.confidence_metrics(result_msa)
    tee("  with-MSA: pTM=$(round(metrics_msa.ptm[1]; digits=3)), time=$(round(t_msa; digits=2))s")

    tee("\n  MSA timing comparison:")
    tee("    without MSA (S=1):   $(round(t_no_msa; digits=2))s")
    tee("    with MSA (S=100):    $(round(t_msa; digits=2))s")
    tee("    ratio: $(round(t_msa / t_no_msa; digits=2))x")
end

# Case template: Fold poly-GLY without and with structural template
# Python's official fold pipeline uses target_templates=true, passing designed structures
# as template data. Without templates, poly-GLY often has 1-2 bond violations (degenerate
# input with no sequence information). With templates, violations drop to 0.
const TEMPLATE_CIF = normpath(joinpath(WORKSPACE_ROOT, "..", "PythonBoltzGen",
    "polyG_official_output", "intermediate_designs_inverse_folded", "polyG_fold_test.cif"))

run_case("case_template_polyG_no_template") do
    # Baseline: fold poly-GLY without template — some violations expected
    result = BoltzGen.fold_from_sequence(fold, "GGGGGGGGGGGGGG"; steps=200, recycles=3, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case_template_polyG_no_template"))
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
    metrics = BoltzGen.confidence_metrics(result)
    tee("  NOTE: poly-GLY without templates may have ~1-2 bond violations (degenerate input)")
end

run_case("case_template_polyG_with_template") do
    isfile(TEMPLATE_CIF) || error("Template CIF not found at $TEMPLATE_CIF")
    result = BoltzGen.fold_from_sequence(fold, "GGGGGGGGGGGGGG";
        steps=200, recycles=3, seed=7, template_paths=[TEMPLATE_CIF])
    BoltzGen.write_outputs(result, joinpath(OUT, "case_template_polyG_with_template"))
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
    metrics = BoltzGen.confidence_metrics(result)
    tee("  Template should reduce/eliminate violations vs no-template baseline")
end

# Release boltz2 conf model
tee("\nReleasing Boltz2 confidence model...")
fold = nothing
GC.gc()
if USE_GPU
    CUDA.reclaim()
end

# ── Phase 3: Boltz2 affinity model ─────────────────────────────────────────────

tee("\n", "#"^60)
tee("  Loading Boltz2 affinity model", USE_GPU ? " (GPU)" : "", "...")
tee("#"^60)
aff = BoltzGen.load_boltz2(; affinity=true, gpu=USE_GPU)
tee("Loaded: ", aff)

# Warmup: compile affinity fold code paths before timing
tee("\n  Warmup: compiling affinity fold code paths...")
warmup_t = @elapsed begin
    BoltzGen.fold_from_sequence(aff, "GGGGG"; steps=2, recycles=1, seed=1)
end
tee("  warmup completed in $(round(warmup_t; digits=1))s")
if USE_GPU; CUDA.synchronize(); end

# Case 05: Fold sequence with affinity (GGGGGGGGGGGGGG)
run_case("case05_fold_sequence_affinity") do
    result = BoltzGen.fold_from_sequence(aff, "GGGGGGGGGGGGGG"; steps=200, recycles=3, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case05_fold_sequence_affinity"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
    metrics = BoltzGen.confidence_metrics(result)
    tee("  affinity metrics: ", pairs(metrics))
end

# Case 06: Fold from structure with affinity (uses case03 output)
run_case("case06_fold_structure_affinity") do
    target_pdb = joinpath(OUT, "case03_design_protein_dna_smiles_atom14.pdb")
    isfile(target_pdb) || error("case03 PDB not found at $target_pdb (did case03 fail?)")
    result = BoltzGen.fold_from_structure(aff, target_pdb; steps=200, recycles=3, seed=7)
    BoltzGen.write_outputs(result, joinpath(OUT, "case06_fold_structure_affinity"))
    check_geometry(result; label="atom14")
    check_geometry(result; label="atom37")
    check_bonds(result)
    check_string_outputs(result)
    metrics = BoltzGen.confidence_metrics(result)
    tee("  affinity metrics: ", pairs(metrics))
end

# Release boltz2 aff model
tee("\nReleasing Boltz2 affinity model...")
aff = nothing
GC.gc()
if USE_GPU
    CUDA.reclaim()
end

# ── Summary ─────────────────────────────────────────────────────────────────────

tee("\n", "="^60)
tee("  REPL API TEST SUMMARY", USE_GPU ? " (GPU)" : " (CPU)")
tee("="^60)
n_pass = count(r -> r[2] == :PASS, results)
n_fail = count(r -> r[2] == :FAIL, results)
tee()
tee("  ", rpad("Case", 40), rpad("Status", 8), "Time (s)")
tee("  ", "-"^40, " ", "-"^6, " ", "-"^10)
for (name, status, detail, elapsed) in results
    if status == :PASS
        tee("  ", rpad(name, 40), rpad("PASS", 8), round(elapsed; digits=2))
    else
        tee("  ", rpad(name, 40), rpad("FAIL", 8), "-")
        for line in split(detail, '\n')[1:min(3, end)]
            tee("        ", line)
        end
    end
end
total_time = sum(r[4] for r in results)
tee()
tee("  Total: ", length(results), "  Pass: ", n_pass, "  Fail: ", n_fail)
tee("  Total time (post-warmup): $(round(total_time; digits=2))s")
tee("  Device: ", USE_GPU ? "GPU" : "CPU")
tee("  Output directory: ", OUT)
tee("  Log file: ", LOG_FILE)
tee("="^60)

close(LOG_IO)

if n_fail > 0
    error("$(n_fail) test case(s) failed.")
end
