include(normpath(joinpath(@__DIR__, "_activate_runfromhere.jl")))

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

function main()
    args = parse_kv_args(ARGS)

    yaml_path = get(
        args,
        "yaml",
        joinpath(WORKSPACE_ROOT, "BoltzGen.jl", "examples", "mixed_protein_dna_smiles_smoke_v1.yaml"),
    )
    steps = parse(Int, get(args, "steps", "60"))
    recycles = parse(Int, get(args, "recycles", "2"))
    seed = parse(Int, get(args, "seed", "7"))
    tag = get(args, "tag", "v1")
    weights = get(
        args,
        "weights",
        "boltzgen1_diverse_state_dict.safetensors",
    )

    out_prefix = joinpath(
        WORKSPACE_ROOT,
        "boltzgen_cache",
        "mixed_protein_dna_smiles_s$(steps)_r$(recycles)_seed$(seed)_$(tag)",
    )
    out_pdb = out_prefix * ".pdb"
    out_pdb37 = out_prefix * "_atom37.pdb"
    out_cif = out_prefix * ".cif"

    run_cmd = `julia --project=$RUNFROMHERE_PROJECT $(joinpath(WORKSPACE_ROOT, "BoltzGen.jl", "scripts", "run_design_from_yaml.jl")) --yaml $yaml_path --steps $(string(steps)) --recycles $(string(recycles)) --seed $(string(seed)) --weights $weights --out-pdb $out_pdb --out-pdb-atom37 $out_pdb37 --out-cif $out_cif`
    geo_cmd = `julia --project=$RUNFROMHERE_PROJECT $(joinpath(WORKSPACE_ROOT, "parity_testing_scripts", "check_output_pdb_geometry.jl")) $out_pdb $out_pdb37`
    dna_cmd = `julia --project=$RUNFROMHERE_PROJECT $(joinpath(WORKSPACE_ROOT, "parity_testing_scripts", "check_dna_geometry.jl")) $out_pdb`

    println("[mixed-smoke] running sample generation")
    run(run_cmd)

    println("[mixed-smoke] running geometry checks")
    run(geo_cmd)

    println("[mixed-smoke] running DNA-specific geometry checks")
    run(dna_cmd)

    println("[mixed-smoke] outputs")
    println("  pdb: ", out_pdb)
    println("  pdb_atom37: ", out_pdb37)
    println("  cif: ", out_cif)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
