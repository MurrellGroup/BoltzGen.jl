import Pkg

const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(WORKSPACE_ROOT, "Onion.jl"))

using Onion

include(normpath(joinpath(@__DIR__, "..", "src", "BoltzGen.jl")))
using .BoltzGen

Onion.bg_set_training!(false)

function parse_kv_args(args)
    out = Dict{String,String}()
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

function parse_index_set(spec::AbstractString, n::Int)
    mask = falses(n)
    isempty(strip(spec)) && return mask
    for part in split(spec, ',')
        s = strip(part)
        isempty(s) && continue
        if occursin('-', s)
            bits = split(s, '-')
            length(bits) == 2 || continue
            a = parse(Int, strip(bits[1]))
            b = parse(Int, strip(bits[2]))
            lo = max(1, min(a, b))
            hi = min(n, max(a, b))
            for i in lo:hi
                mask[i] = true
            end
        else
            idx = parse(Int, s)
            if 1 <= idx <= n
                mask[idx] = true
            end
        end
    end
    return mask
end

function parse_token_labels(spec::AbstractString, n::Int, default_value::Int, map::Dict{String,Int})
    v = fill(default_value, n)
    isempty(strip(spec)) && return v
    for item in split(spec, ',')
        s = strip(item)
        isempty(s) && continue
        bits = split(s, ':')
        length(bits) == 2 || continue
        idx = parse(Int, strip(bits[1]))
        label = uppercase(strip(bits[2]))
        if 1 <= idx <= n && haskey(map, label)
            v[idx] = map[label]
        end
    end
    return v
end

function parse_structure_groups(spec::AbstractString, n::Int)
    v = fill(0, n)
    isempty(strip(spec)) && return v
    for item in split(spec, ',')
        s = strip(item)
        isempty(s) && continue
        bits = split(s, ':')
        length(bits) == 2 || continue
        idx = parse(Int, strip(bits[1]))
        grp = parse(Int, strip(bits[2]))
        if 1 <= idx <= n
            v[idx] = grp
        end
    end
    return v
end

function parse_bonds(spec::AbstractString)
    out = NTuple{3,Int}[]
    isempty(strip(spec)) && return out
    for item in split(spec, ',')
        s = strip(item)
        isempty(s) && continue
        bits = split(s, ':')
        length(bits) == 3 || continue
        i = parse(Int, strip(bits[1]))
        j = parse(Int, strip(bits[2]))
        bt = parse(Int, strip(bits[3]))
        push!(out, (i, j, bt))
    end
    return out
end

function main()
    args = parse_kv_args(ARGS)

    chain_type = uppercase(get(args, "chain-type", "PROTEIN"))
    sequence = get(args, "sequence", "")
    length_arg = parse(Int, get(args, "length", "13"))

    residues = if !isempty(sequence)
        BoltzGen.tokens_from_sequence(sequence; chain_type=chain_type)
    else
        if chain_type == "PROTEIN"
            fill("GLY", length_arg)
        elseif chain_type == "DNA"
            fill("DN", length_arg)
        elseif chain_type == "RNA"
            fill("N", length_arg)
        else
            error("Unsupported chain-type: $chain_type")
        end
    end

    T = length(residues)
    steps = parse(Int, get(args, "steps", "100"))
    recycles = parse(Int, get(args, "recycles", "3"))

    default_base = joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_design_len$(T)_julia")
    out_pdb = get(args, "out-pdb", default_base * ".pdb")
    out_pdb37 = get(args, "out-pdb-atom37", default_base * "_atom37.pdb")
    out_cif = get(args, "out-cif", default_base * ".cif")
    weights_path = get(args, "weights", joinpath(WORKSPACE_ROOT, "boltzgen_cache", "boltzgen1_diverse_state_dict.safetensors"))

    mol_type_id = BoltzGen.chain_type_ids[chain_type]
    mol_types = fill(mol_type_id, T)

    design_mask = if haskey(args, "design-mask")
        parse_index_set(args["design-mask"], T)
    else
        trues(T)
    end

    target_msa_mask = if haskey(args, "target-msa-mask")
        parse_index_set(args["target-msa-mask"], T)
    else
        falses(T)
    end

    affinity_token_mask = if haskey(args, "affinity-mask")
        parse_index_set(args["affinity-mask"], T)
    else
        falses(T)
    end

    binding_labels = parse_token_labels(
        get(args, "binding", ""),
        T,
        BoltzGen.binding_type_ids["UNSPECIFIED"],
        Dict(k => v for (k, v) in BoltzGen.binding_type_ids),
    )

    ss_labels = parse_token_labels(
        get(args, "ss", ""),
        T,
        BoltzGen.ss_type_ids["UNSPECIFIED"],
        Dict(k => v for (k, v) in BoltzGen.ss_type_ids),
    )

    structure_groups = parse_structure_groups(get(args, "structure-group", ""), T)
    bonds = parse_bonds(get(args, "bonds", ""))

    model, _, missing = BoltzGen.load_model_from_safetensors(weights_path)
    if !isempty(missing)
        println("Unmapped state keys: ", length(missing))
    end

    feats = BoltzGen.build_design_features(
        residues;
        mol_types=mol_types,
        design_mask=design_mask,
        binding_type=binding_labels,
        ss_type=ss_labels,
        structure_group=structure_groups,
        target_msa_mask=target_msa_mask,
        affinity_token_mask=affinity_token_mask,
        bonds=bonds,
        batch=1,
    )

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

    # Restore original (pre-mask) atom names/elements for non-protein or non-designed tokens.
    atom_to_token = feats_masked["atom_to_token"][:, :, 1]
    atom_pad_mask = feats_masked["atom_pad_mask"][:, 1]
    design_m = feats["design_mask"][:, 1]
    mol_type = feats["mol_type"][:, 1]
    for m in 1:size(atom_to_token, 1)
        atom_pad_mask[m] > 0.5 || continue
        t = argmax(view(atom_to_token, m, :))
        is_designed_protein = (design_m[t] > 0) && (mol_type[t] == BoltzGen.chain_type_ids["PROTEIN"])
        if !is_designed_protein
            feats_out["ref_atom_name_chars"][:, :, m, 1] .= feats["ref_atom_name_chars"][:, :, m, 1]
            feats_out["ref_element"][:, m, 1] .= feats["ref_element"][:, m, 1]
        end
    end
    for t in 1:size(feats["res_type"], 2)
        is_designed_protein = (design_m[t] > 0) && (mol_type[t] == BoltzGen.chain_type_ids["PROTEIN"])
        if !is_designed_protein
            feats_out["res_type"][:, t, 1] .= feats["res_type"][:, t, 1]
        end
    end

    mkpath(dirname(out_pdb))
    mkpath(dirname(out_pdb37))
    mkpath(dirname(out_cif))

    BoltzGen.write_pdb(out_pdb, feats_out, coords; batch=1)
    BoltzGen.write_pdb_atom37(out_pdb37, feats_out, coords; batch=1)
    BoltzGen.write_mmcif(out_cif, feats_out, coords; batch=1)

    println("Wrote PDB (atom14-like): ", out_pdb)
    println("Wrote PDB (atom37 mapped): ", out_pdb37)
    println("Wrote mmCIF (atom37 mapped): ", out_cif)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
