import Pkg

const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(WORKSPACE_ROOT, "Onion.jl"))

using Onion

include(normpath(joinpath(@__DIR__, "..", "src", "BoltzGen.jl")))
using .BoltzGen

Onion.bg_set_training!(false)

function main()
    token_len = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 13
    num_sampling_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
    recycling_steps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    out_pdb = length(ARGS) >= 4 ? ARGS[4] : joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_denovo_len$(token_len)_julia.pdb")
    weights_path = length(ARGS) >= 5 ? ARGS[5] : joinpath(WORKSPACE_ROOT, "boltzgen_cache", "boltzgen1_diverse_state_dict.safetensors")

    model, _, missing = BoltzGen.load_model_from_safetensors(weights_path)
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
    println("Wrote PDB: ", out_pdb)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
