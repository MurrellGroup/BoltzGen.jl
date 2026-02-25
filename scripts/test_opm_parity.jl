#!/usr/bin/env julia
#
# Test parity between optimized (batched_mul) and original (mapreduce)
# OuterProductMean implementations.
#
# Usage:
#   julia --project=runfromhere BoltzGen.jl/scripts/test_opm_parity.jl
#   julia --project=runfromhere BoltzGen.jl/scripts/test_opm_parity.jl --gpu
#

using Random

const USE_GPU = "--gpu" in ARGS

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

println("\n", "="^60)
println("  OuterProductMean: Parity Test (batched_mul vs mapreduce)")
println("  Device: ", USE_GPU ? "GPU" : "CPU")
println("="^60)

# Test multiple configurations
configs = [
    (c_in=64, c_hidden=32, c_out=128, S=1, N=14, B=1, name="small S=1"),
    (c_in=64, c_hidden=32, c_out=128, S=4, N=14, B=1, name="small S=4"),
    (c_in=64, c_hidden=32, c_out=128, S=1, N=50, B=1, name="medium N=50"),
    (c_in=64, c_hidden=32, c_out=128, S=8, N=50, B=1, name="medium S=8 N=50"),
    (c_in=64, c_hidden=32, c_out=128, S=1, N=120, B=1, name="large N=120 (antibody)"),
    (c_in=64, c_hidden=32, c_out=128, S=4, N=120, B=1, name="large S=4 N=120"),
]

all_pass = true

for cfg in configs
    println("\n--- $(cfg.name): c_in=$(cfg.c_in), c_hidden=$(cfg.c_hidden), c_out=$(cfg.c_out), S=$(cfg.S), N=$(cfg.N), B=$(cfg.B) ---")

    Random.seed!(42)
    layer = Onion.OuterProductMean(cfg.c_in, cfg.c_hidden, cfg.c_out)

    m_cpu = randn(Float32, cfg.c_in, cfg.S, cfg.N, cfg.B)
    mask_cpu = ones(Float32, cfg.S, cfg.N, cfg.B)
    # Set some mask values to 0 for realism
    if cfg.S > 1
        mask_cpu[end, :, :] .= 0f0
    end

    if USE_GPU
        layer_gpu = Onion.Flux.gpu(layer)
        m_in = CUDA.cu(m_cpu)
        mask_in = CUDA.cu(mask_cpu)

        # Run optimized (current default)
        out_new = layer_gpu(m_in, mask_in)
        CUDA.synchronize()
        out_new_cpu = Array(out_new)

        # Run original mapreduce
        out_old = Onion._opm_forward_mapreduce(layer_gpu, m_in, mask_in)
        CUDA.synchronize()
        out_old_cpu = Array(out_old)
    else
        m_in = m_cpu
        mask_in = mask_cpu

        # Run optimized (current default)
        out_new = layer(m_in, mask_in)
        out_new_cpu = out_new

        # Run original mapreduce
        out_old = Onion._opm_forward_mapreduce(layer, m_in, mask_in)
        out_old_cpu = out_old
    end

    # Compare
    max_diff = maximum(abs.(out_new_cpu .- out_old_cpu))
    mean_diff = sum(abs.(out_new_cpu .- out_old_cpu)) / length(out_new_cpu)
    rel_scale = max(maximum(abs.(out_old_cpu)), 1f-8)
    max_rel = max_diff / rel_scale

    pass = max_diff < 1f-4  # Allow small floating point differences
    status = pass ? "PASS" : "FAIL"
    global all_pass = all_pass && pass

    println("  output shape: ", size(out_new_cpu))
    println("  max abs diff: ", max_diff)
    println("  mean abs diff: ", mean_diff)
    println("  max relative diff: ", max_rel)
    println("  [$status]")

    # Timing comparison (3 runs each, take min)
    if USE_GPU
        CUDA.synchronize()
    end

    times_new = Float64[]
    times_old = Float64[]
    for _ in 1:3
        t = @elapsed begin
            layer_fn = USE_GPU ? layer_gpu : layer
            _ = layer_fn(m_in, mask_in)
            USE_GPU && CUDA.synchronize()
        end
        push!(times_new, t)
    end
    for _ in 1:3
        t = @elapsed begin
            layer_fn = USE_GPU ? layer_gpu : layer
            _ = Onion._opm_forward_mapreduce(layer_fn, m_in, mask_in)
            USE_GPU && CUDA.synchronize()
        end
        push!(times_old, t)
    end

    t_new = minimum(times_new)
    t_old = minimum(times_old)
    speedup = t_old / max(t_new, 1e-9)
    println("  timing: batched_mul=$(round(t_new*1000; digits=2))ms, mapreduce=$(round(t_old*1000; digits=2))ms, speedup=$(round(speedup; digits=2))x")
end

println("\n", "="^60)
if all_pass
    println("  ALL PARITY TESTS PASSED")
else
    println("  SOME TESTS FAILED")
end
println("="^60)

all_pass || error("Parity test failed!")
