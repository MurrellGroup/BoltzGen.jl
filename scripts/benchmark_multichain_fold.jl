#!/usr/bin/env julia
#
# Benchmark multi-chain antibody folding: 1, 2, 3, 6 chains.
# Excludes compilation time by running a warmup fold first.
#
# Usage:
#   julia --project=runfromhere BoltzGen.jl/scripts/benchmark_multichain_fold.jl
#   julia --project=runfromhere BoltzGen.jl/scripts/benchmark_multichain_fold.jl --gpu
#

using Random

# Parse --gpu flag
const USE_GPU = "--gpu" in ARGS
non_flag_args = filter(a -> a != "--gpu", ARGS)

if USE_GPU
    println("GPU mode enabled — loading CUDA and cuDNN...")
    using CUDA
    using cuDNN
    println("CUDA functional: ", CUDA.functional())
    CUDA.functional() || error("CUDA is not functional.")
    println("GPU device: ", CUDA.device())
end

include(normpath(joinpath(@__DIR__, "..", "src", "BoltzGen.jl")))
using .BoltzGen
using Onion

Onion.bg_set_training!(false)

const STEPS = 200
const RECYCLES = 3

# ── Antibody sequences ────────────────────────────────────────────────────────
# Trastuzumab (Herceptin)
const TRASTUZUMAB_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
const TRASTUZUMAB_VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

# Pertuzumab
const PERTUZUMAB_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEWVAGITPAGGYTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
const PERTUZUMAB_VL = "DIQMTQSPSSLSASVGDRVTITCKASQDVSIGVAWYQQKPGKAPKLLIYSASYRYTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYYIYPYTFGQGTKVEIK"

# Bevacizumab (Avastin)
const BEVACIZUMAB_VH = "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS"
const BEVACIZUMAB_VL = "DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"

# ── Benchmark configurations ──────────────────────────────────────────────────

const CONFIGS = [
    (name="1-chain (VH only)",
     sequences=[TRASTUZUMAB_VH]),

    (name="2-chain (VH+VL)",
     sequences=[TRASTUZUMAB_VH, TRASTUZUMAB_VL]),

    (name="3-chain (VH+VL+VH)",
     sequences=[TRASTUZUMAB_VH, TRASTUZUMAB_VL, PERTUZUMAB_VH]),

    (name="6-chain (3x VH+VL)",
     sequences=[TRASTUZUMAB_VH, TRASTUZUMAB_VL,
                PERTUZUMAB_VH, PERTUZUMAB_VL,
                BEVACIZUMAB_VH, BEVACIZUMAB_VL]),
]

# ── Load model ────────────────────────────────────────────────────────────────

println("\n", "#"^60)
println("  Loading Boltz2 confidence model", USE_GPU ? " (GPU)" : "", "...")
println("#"^60)
fold = BoltzGen.load_boltz2(; gpu=USE_GPU)
println("Loaded: ", fold)

# ── Warmup (exclude compilation time) ─────────────────────────────────────────

println("\n", "="^60)
println("  WARMUP: Folding short poly-G sequence (compile time excluded)")
println("="^60)
warmup_t = @elapsed begin
    BoltzGen.fold_from_sequence(fold, "GGGGGGGGGG"; steps=5, recycles=1, seed=1)
end
println("  warmup completed in $(round(warmup_t; digits=1))s")

# Also warmup multi-chain path
warmup_t2 = @elapsed begin
    BoltzGen.fold_from_sequences(fold, ["GGGGG", "GGGGG"]; steps=5, recycles=1, seed=1)
end
println("  multi-chain warmup completed in $(round(warmup_t2; digits=1))s")

if USE_GPU
    CUDA.synchronize()
end

# ── Benchmark ─────────────────────────────────────────────────────────────────

println("\n", "#"^60)
println("  MULTI-CHAIN FOLDING BENCHMARK")
println("  steps=$STEPS, recycles=$RECYCLES, device=", USE_GPU ? "GPU" : "CPU")
println("#"^60)

results = NamedTuple[]

for cfg in CONFIGS
    n_chains = length(cfg.sequences)
    total_residues = sum(length.(cfg.sequences))
    println("\n", "-"^60)
    println("  $(cfg.name): $n_chains chain(s), $total_residues total residues")
    println("-"^60)

    if USE_GPU
        CUDA.synchronize()
    end
    GC.gc()

    t_start = time()
    result = if n_chains == 1
        BoltzGen.fold_from_sequence(fold, cfg.sequences[1]; steps=STEPS, recycles=RECYCLES, seed=7)
    else
        BoltzGen.fold_from_sequences(fold, cfg.sequences; steps=STEPS, recycles=RECYCLES, seed=7)
    end
    if USE_GPU
        CUDA.synchronize()
    end
    elapsed = time() - t_start

    # Geometry check
    stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"]; batch=1)
    metrics = BoltzGen.confidence_metrics(result)

    push!(results, (
        name=cfg.name,
        n_chains=n_chains,
        total_residues=total_residues,
        elapsed=elapsed,
        n_atoms=stats.n_atoms,
        ptm=metrics.ptm[1],
        iptm=metrics.iptm[1],
    ))

    println("  time: $(round(elapsed; digits=2))s")
    println("  atoms: $(stats.n_atoms)")
    println("  pTM: $(round(metrics.ptm[1]; digits=3)), ipTM: $(round(metrics.iptm[1]; digits=3))")
    println("  geometry: max_abs=$(round(stats.max_abs; digits=1)), min_nn=$(round(stats.min_nearest_neighbor; digits=3))")
end

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n", "="^60)
println("  BENCHMARK SUMMARY (device=", USE_GPU ? "GPU" : "CPU", ")")
println("="^60)
println()
println("  ", rpad("Config", 25), rpad("Chains", 8), rpad("Residues", 10),
    rpad("Atoms", 8), rpad("Time (s)", 10), rpad("pTM", 8), "ipTM")
println("  ", "-"^25, " ", "-"^7, " ", "-"^9, " ", "-"^7, " ", "-"^9, " ", "-"^7, " ", "-"^7)
for r in results
    println("  ",
        rpad(r.name, 25),
        rpad(r.n_chains, 8),
        rpad(r.total_residues, 10),
        rpad(r.n_atoms, 8),
        rpad(round(r.elapsed; digits=2), 10),
        rpad(round(r.ptm; digits=3), 8),
        round(r.iptm; digits=3))
end

# Scaling analysis
if length(results) >= 2
    println()
    base = results[1].elapsed
    for r in results[2:end]
        println("  $(r.name): $(round(r.elapsed / base; digits=2))x vs single-chain")
    end
end

println()
println("="^60)
