using CUDA
using cuDNN
using Onion
using SafeTensors
using Random

include(normpath(joinpath(@__DIR__, "..", "src", "BoltzGen.jl")))
using .BoltzGen

Onion.bg_set_training!(false)

function require_sampling_checkpoint!(weights_spec::AbstractString; requires_design_conditioning::Bool=false)
    weights_path = BoltzGen.resolve_weights_path(weights_spec)
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
    return weights_path
end

function parse_model_family(spec::AbstractString)
    fam = lowercase(strip(String(spec)))
    fam in ("boltzgen1", "boltz2") || error("Unsupported model family '$spec' (expected boltzgen1 or boltz2)")
    return fam
end

function main()
    token_len = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 13
    num_sampling_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
    recycling_steps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    out_pdb = length(ARGS) >= 4 ? ARGS[4] : joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_denovo_len$(token_len)_julia.pdb")
    out_pdb37 = replace(out_pdb, ".pdb" => "_atom37.pdb")
    out_pdb37 == out_pdb && (out_pdb37 = out_pdb * "_atom37.pdb")
    weights_spec = length(ARGS) >= 5 ? ARGS[5] : "boltzgen1_diverse_state_dict.safetensors"
    model_family = length(ARGS) >= 10 ? parse_model_family(ARGS[10]) : "boltzgen1"
    if model_family == "boltz2"
        error("Boltz2 mode is folding-only and not supported by run_denovo_sample.jl.")
    end
    weights_path = require_sampling_checkpoint!(weights_spec; requires_design_conditioning=true)
    with_confidence = length(ARGS) >= 6 ? (ARGS[6] == "1") : false
    with_affinity = length(ARGS) >= 7 ? (ARGS[7] == "1") : false
    out_heads = length(ARGS) >= 8 ? ARGS[8] : ""
    seed = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 1
    Random.seed!(seed)
    println("Using seed: ", seed)

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
        step_scale=1.8f0,
        noise_scale=0.95f0,
        inference_logging=false,
    )

    coords = out["sample_atom_coords"]
    feats_out = BoltzGen.postprocess_atom14(feats_masked, coords)

    mkpath(dirname(out_pdb))
    mkpath(dirname(out_pdb37))
    BoltzGen.write_pdb(out_pdb, feats_out, coords; batch=1)
    BoltzGen.write_pdb_atom37(out_pdb37, feats_out, coords; batch=1)

    # Bond length validation
    bond_stats = BoltzGen.check_bond_lengths(feats_out, coords; batch=1)
    BoltzGen.print_bond_length_report(bond_stats)

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
    println("Wrote PDB (atom37 mapped): ", out_pdb37)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
