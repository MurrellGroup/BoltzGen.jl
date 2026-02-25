#!/usr/bin/env julia
#
# Test parity between optimized (single batched_mul) and original (map-loop)
# PairWeightedAveraging implementations.
#
# Usage:
#   julia --project=runfromhere BoltzGen.jl/scripts/test_pwa_parity.jl
#   julia --project=runfromhere BoltzGen.jl/scripts/test_pwa_parity.jl --gpu
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
println("  PairWeightedAveraging: Parity Test (batched_mul vs map-loop)")
println("  Device: ", USE_GPU ? "GPU" : "CPU")
println("="^60)

# Test multiple configurations
configs = [
    (c_m=64, c_z=128, c_h=32, num_heads=4, S=1, N=14, B=1, name="small S=1"),
    (c_m=64, c_z=128, c_h=32, num_heads=4, S=4, N=14, B=1, name="small S=4"),
    (c_m=64, c_z=128, c_h=32, num_heads=4, S=16, N=50, B=1, name="medium S=16 N=50"),
    (c_m=64, c_z=128, c_h=32, num_heads=4, S=50, N=50, B=1, name="large S=50 N=50"),
    (c_m=64, c_z=128, c_h=32, num_heads=4, S=100, N=120, B=1, name="MSA-scale S=100 N=120"),
]

all_pass = true

for cfg in configs
    println("\n--- $(cfg.name): c_m=$(cfg.c_m), c_z=$(cfg.c_z), c_h=$(cfg.c_h), heads=$(cfg.num_heads), S=$(cfg.S), N=$(cfg.N), B=$(cfg.B) ---")

    Random.seed!(42)
    layer = Onion.PairWeightedAveraging(cfg.c_m, cfg.c_z, cfg.c_h, cfg.num_heads)

    m_cpu = randn(Float32, cfg.c_m, cfg.S, cfg.N, cfg.B)
    z_cpu = randn(Float32, cfg.c_z, cfg.N, cfg.N, cfg.B)
    mask_cpu = ones(Float32, cfg.N, cfg.N, cfg.B)

    if USE_GPU
        layer_gpu = Onion.Flux.gpu(layer)
        m_in = CUDA.cu(m_cpu)
        z_in = CUDA.cu(z_cpu)
        mask_in = CUDA.cu(mask_cpu)

        # Run optimized (current default)
        out_new = layer_gpu(m_in, z_in, mask_in)
        CUDA.synchronize()
        out_new_cpu = Array(out_new)

        # Run original map-loop
        out_old = Onion._pwa_forward_maploop(layer_gpu, m_in, z_in, mask_in)
        CUDA.synchronize()
        out_old_cpu = Array(out_old)
    else
        m_in = m_cpu
        z_in = z_cpu
        mask_in = mask_cpu

        # Run optimized (current default)
        out_new = layer(m_in, z_in, mask_in)
        out_new_cpu = out_new

        # Run original map-loop
        out_old = Onion._pwa_forward_maploop(layer, m_in, z_in, mask_in)
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
            if USE_GPU
                _ = layer_gpu(m_in, z_in, mask_in)
                CUDA.synchronize()
            else
                _ = layer(m_in, z_in, mask_in)
            end
        end
        push!(times_new, t)
    end
    for _ in 1:3
        t = @elapsed begin
            if USE_GPU
                _ = Onion._pwa_forward_maploop(layer_gpu, m_in, z_in, mask_in)
                CUDA.synchronize()
            else
                _ = Onion._pwa_forward_maploop(layer, m_in, z_in, mask_in)
            end
        end
        push!(times_old, t)
    end

    t_new = minimum(times_new)
    t_old = minimum(times_old)
    speedup = t_old / max(t_new, 1e-9)
    println("  timing: batched_mul=$(round(t_new*1000; digits=2))ms, map-loop=$(round(t_old*1000; digits=2))ms, speedup=$(round(speedup; digits=2))x")
end

println("\n", "="^60)
if all_pass
    println("  ALL PARITY TESTS PASSED")
else
    println("  SOME TESTS FAILED")
end
println("="^60)

all_pass || error("Parity test failed!")
