import Pkg

const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(WORKSPACE_ROOT, "Onion.jl"))

using Onion
using Random

include(normpath(joinpath(@__DIR__, "..", "src", "BoltzGen.jl")))
using .BoltzGen

Onion.bg_set_training!(false)

function parse_kv_args(args)
    out = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if startswith(key, "--")
            key = key[3:end]
            if i < length(args) && !startswith(args[i + 1], "--")
                out[key] = args[i + 1]
                i += 2
            else
                out[key] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return out
end

function py_env(py_path::AbstractString)
    pp = joinpath(WORKSPACE_ROOT, "boltzgen", "src")
    if haskey(ENV, "PYTHONPATH") && !isempty(ENV["PYTHONPATH"])
        return ["PYTHONPATH" => pp * ":" * ENV["PYTHONPATH"]]
    end
    return ["PYTHONPATH" => pp]
end

function main()
    args = parse_kv_args(ARGS)

    yaml_path = get(args, "yaml", "")
    isempty(yaml_path) && error("Missing --yaml <path-to-boltzgen-yaml>")

    if haskey(args, "seed")
        seed = parse(Int, args["seed"])
        Random.seed!(seed)
        println("Using seed: ", seed)
    end

    with_confidence = get(args, "with-confidence", "false") == "true"
    with_affinity = get(args, "with-affinity", "false") == "true"

    steps = parse(Int, get(args, "steps", "100"))
    recycles = parse(Int, get(args, "recycles", "3"))

    weights_path = get(args, "weights", joinpath(WORKSPACE_ROOT, "boltzgen_cache", "boltzgen1_diverse_state_dict.safetensors"))

    stem = splitext(basename(yaml_path))[1]
    default_prefix = joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_from_yaml_" * stem)
    out_pdb = get(args, "out-pdb", default_prefix * ".pdb")
    out_pdb37 = get(args, "out-pdb-atom37", default_prefix * "_atom37.pdb")
    out_cif = get(args, "out-cif", default_prefix * ".cif")
    out_heads = get(args, "out-heads", "")

    features_npz = get(args, "features-npz", default_prefix * "_features.npz")
    moldir = get(args, "moldir", joinpath(WORKSPACE_ROOT, "boltzgen_cache", "mols"))

    py = get(args, "python", joinpath(WORKSPACE_ROOT, "boltzgen", ".venv", "bin", "python"))
    py_script = joinpath(@__DIR__, "export_yaml_features.py")
    py_cmd = Cmd([
        py,
        py_script,
        "--yaml", yaml_path,
        "--moldir", moldir,
        "--out", features_npz,
        with_affinity ? "--compute-affinity" : "",
    ])
    py_cmd = Cmd(filter(!isempty, collect(py_cmd.exec)))

    mkpath(dirname(features_npz))
    mkpath(dirname(out_pdb))
    mkpath(dirname(out_pdb37))
    mkpath(dirname(out_cif))

    run(setenv(py_cmd, py_env(py)...))

    feats = BoltzGen.load_python_feature_npz(features_npz)

    model, _, missing = BoltzGen.load_model_from_safetensors(
        weights_path;
        confidence_prediction=with_confidence,
        affinity_prediction=with_affinity,
    )
    if !isempty(missing)
        println("Unmapped state keys: ", length(missing))
    end

    feats_masked = BoltzGen.boltz_masker(feats; mask=true, mask_backbone=false)
    out = BoltzGen.boltz_forward(
        model,
        feats_masked;
        recycling_steps=recycles,
        num_sampling_steps=steps,
        diffusion_samples=1,
        inference_logging=false,
    )

    coords = out["sample_atom_coords"]
    feats_out = BoltzGen.postprocess_atom14(feats_masked, coords)

    BoltzGen.write_pdb(out_pdb, feats_out, coords; batch=1)
    BoltzGen.write_pdb_atom37(out_pdb37, feats_out, coords; batch=1)
    BoltzGen.write_mmcif(out_cif, feats_out, coords; batch=1)

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

    println("Wrote features NPZ: ", features_npz)
    println("Wrote PDB (atom14-like): ", out_pdb)
    println("Wrote PDB (atom37 mapped): ", out_pdb37)
    println("Wrote mmCIF (atom37 mapped): ", out_cif)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
