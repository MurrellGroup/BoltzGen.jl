import Pkg

const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(WORKSPACE_ROOT, "Onion.jl"))

using Onion
using SafeTensors
using Random

include(normpath(joinpath(@__DIR__, "..", "src", "BoltzGen.jl")))
using .BoltzGen

Onion.bg_set_training!(false)

function require_sampling_checkpoint!(weights_path::AbstractString; requires_design_conditioning::Bool=false)
    state = SafeTensors.load_safetensors(weights_path)
    has_token_transformer = any(startswith(k, "structure_module.score_model.token_transformer_layers.0.layers.") for k in keys(state)) ||
        any(startswith(k, "structure_module.score_model.token_transformer.layers.") for k in keys(state))
    if !has_token_transformer
        error(
            "Checkpoint lacks structure diffusion token-transformer weights and cannot be used for sampling: $weights_path. " *
            "Use a full structure checkpoint (e.g. boltzgen1_diverse) or a merged base+head checkpoint.",
        )
    end
    if requires_design_conditioning
        has_design_conditioning = haskey(state, "input_embedder.design_mask_conditioning_init.weight")
        if !has_design_conditioning
            error(
                "Checkpoint is missing design-conditioning weights and cannot perform de novo design sampling: $weights_path. " *
                "Use a design checkpoint (e.g. boltzgen1_diverse or boltzgen1_adherence).",
            )
        end
    end
end

function main()
    token_len = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 13
    num_sampling_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
    recycling_steps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    out_pdb = length(ARGS) >= 4 ? ARGS[4] : joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_denovo_len$(token_len)_julia.pdb")
    weights_path = length(ARGS) >= 5 ? ARGS[5] : joinpath(WORKSPACE_ROOT, "boltzgen_cache", "boltzgen1_diverse_state_dict.safetensors")
    require_sampling_checkpoint!(weights_path; requires_design_conditioning=true)
    with_confidence = length(ARGS) >= 6 ? (ARGS[6] == "1") : false
    with_affinity = length(ARGS) >= 7 ? (ARGS[7] == "1") : false
    out_heads = length(ARGS) >= 8 ? ARGS[8] : ""
    if length(ARGS) >= 9
        seed = parse(Int, ARGS[9])
        Random.seed!(seed)
        println("Using seed: ", seed)
    end

    model, _, missing = BoltzGen.load_model_from_safetensors(
        weights_path;
        confidence_prediction=with_confidence,
        affinity_prediction=with_affinity,
    )
    if !isempty(missing)
        println("Unmapped state keys: ", length(missing))
    end

    feats = BoltzGen.build_denovo_atom14_features(token_len)
    feats_masked = BoltzGen.boltz_masker(feats; mask=true, mask_backbone=false)

    out = BoltzGen.boltz_forward(
        model,
        feats_masked;
        recycling_steps=recycling_steps,
        num_sampling_steps=num_sampling_steps,
        diffusion_samples=1,
        inference_logging=false,
    )

    coords = out["sample_atom_coords"]
    feats_out = BoltzGen.postprocess_atom14(feats_masked, coords)
    mkpath(dirname(out_pdb))
    BoltzGen.write_pdb(out_pdb, feats_out, coords; batch=1)
    if !isempty(out_heads)
        mkpath(dirname(out_heads))
        open(out_heads, "w") do io
            for k in (
                "ptm", "iptm", "complex_plddt", "complex_iplddt",
                "affinity_pred_value", "affinity_probability_binary",
                "affinity_pred_value1", "affinity_probability_binary1",
                "affinity_pred_value2", "affinity_probability_binary2",
            )
                if haskey(out, k)
                    v = vec(Float32.(out[k]))
                    println(io, k, "=", join(v, ","))
                end
            end
        end
        println("Wrote head summary: ", out_heads)
    end
    println("Wrote PDB: ", out_pdb)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
