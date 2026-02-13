include(normpath(joinpath(@__DIR__, "_activate_runfromhere.jl")))

using Onion
using SafeTensors
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

function parse_string_list(spec::AbstractString)
    isempty(strip(spec)) && return String[]
    return [String(strip(s)) for s in split(spec, ',') if !isempty(strip(s))]
end

function parse_model_family(spec::AbstractString)
    fam = lowercase(strip(String(spec)))
    fam in ("boltzgen1", "boltz2") || error("Unsupported --model-family '$spec' (expected boltzgen1 or boltz2)")
    return fam
end

function default_weights_for_family(model_family::AbstractString, with_affinity::Bool)
    return BoltzGen.default_weights_filename(model_family, with_affinity)
end

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
    # Return the user-facing HF spec, not the resolved local cache path.
    return strip(String(weights_spec))
end

function main()
    args = parse_kv_args(ARGS)

    yaml_path = get(args, "yaml", "")
    isempty(yaml_path) && error("Missing --yaml <path-to-boltzgen-yaml>")

    seed = haskey(args, "seed") ? parse(Int, args["seed"]) : 1
    Random.seed!(seed)
    println("Using seed: ", seed)

    model_family = parse_model_family(get(args, "model-family", "boltzgen1"))
    with_affinity = get(args, "with-affinity", "false") == "true"
    with_confidence_default = model_family == "boltz2" ? "true" : "false"
    with_confidence = get(args, "with-confidence", with_confidence_default) == "true"
    include_nonpolymer_default = "true"
    include_nonpolymer = get(args, "include-nonpolymer", include_nonpolymer_default) == "true"

    out_heads = get(args, "out-heads", "")
    steps = parse(Int, get(args, "steps", "100"))
    recycles = parse(Int, get(args, "recycles", "3"))

    weights_spec = get(args, "weights", default_weights_for_family(model_family, with_affinity))

    stem = splitext(basename(yaml_path))[1]
    out_prefix_default = joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_from_yaml_" * stem)
    out_pdb = get(args, "out-pdb", out_prefix_default * ".pdb")
    out_pdb37 = get(args, "out-pdb-atom37", out_prefix_default * "_atom37.pdb")
    out_cif = get(args, "out-cif", out_prefix_default * ".cif")

    msa_file = get(args, "msa-file", "")
    msa_max_rows = haskey(args, "msa-max-rows") ? parse(Int, args["msa-max-rows"]) : nothing
    template_paths = parse_string_list(get(args, "template-paths", ""))
    template_max_count = haskey(args, "template-max-count") ? parse(Int, args["template-max-count"]) : nothing
    template_chains = haskey(args, "template-chains") ? parse_string_list(args["template-chains"]) : nothing

    parsed = BoltzGen.parse_design_yaml(yaml_path; include_nonpolymer=include_nonpolymer)
    println(
        "[progress] case-ready: yaml parsed (tokens=",
        length(parsed.residue_tokens),
        ", yaml=",
        yaml_path,
        ")",
    )
    msa_sequences = if isempty(msa_file)
        parsed.msa_sequences
    else
        BoltzGen.load_msa_sequences(msa_file; max_rows=msa_max_rows)
    end
    msa_paired_rows = isempty(msa_file) ? get(parsed, :msa_paired_rows, nothing) : nothing
    msa_has_deletion_rows = isempty(msa_file) ? get(parsed, :msa_has_deletion_rows, nothing) : nothing
    msa_deletion_value_rows = isempty(msa_file) ? get(parsed, :msa_deletion_value_rows, nothing) : nothing
    if isempty(msa_file) && parsed.msa_path !== nothing
        println("Using YAML MSA file: ", parsed.msa_path)
    elseif isempty(msa_file) && haskey(parsed, :msa_paths) && length(parsed.msa_paths) > 1
        println("Using merged YAML MSA files: ", join(parsed.msa_paths, ", "))
    end

    affinity_token_mask = if haskey(args, "affinity-mask")
        mask = falses(length(parsed.residue_tokens))
        for item in split(args["affinity-mask"], ',')
            s = strip(item)
            isempty(s) && continue
            idx = parse(Int, s)
            if 1 <= idx <= length(mask)
                mask[idx] = true
            end
        end
        mask
    elseif with_affinity
        [parsed.mol_types[i] == BoltzGen.chain_type_ids["NONPOLYMER"] for i in eachindex(parsed.mol_types)]
    else
        falses(length(parsed.residue_tokens))
    end
    affinity_mw = haskey(args, "affinity-mw") ? Float32(parse(Float64, args["affinity-mw"])) : 0f0

    if model_family == "boltz2" && any(parsed.design_mask)
        error("Boltz2 mode is folding-only in this script; YAML specs with designed residues are not supported.")
    end

    weights_path = require_sampling_checkpoint!(weights_spec; requires_design_conditioning=any(parsed.design_mask))

    model, _, missing = BoltzGen.load_model_from_safetensors(
        weights_path;
        confidence_prediction=with_confidence,
        affinity_prediction=with_affinity,
    )
    if !isempty(missing)
        println("Unmapped state keys: ", length(missing))
    end
    println("[progress] case-ready: model loaded (weights=", basename(weights_path), ")")

    feats = BoltzGen.build_design_features(
        parsed.residue_tokens;
        mol_types=parsed.mol_types,
        asym_ids=parsed.asym_ids,
        entity_ids=parsed.entity_ids,
        sym_ids=parsed.sym_ids,
        cyclic_period=parsed.cyclic_period,
        residue_indices=parsed.residue_indices,
        design_mask=parsed.design_mask,
        binding_type=parsed.binding_type,
        ss_type=parsed.ss_type,
        structure_group=parsed.structure_group,
        target_msa_mask=parsed.target_msa_mask,
        msa_sequences=msa_sequences,
        msa_paired_rows=msa_paired_rows,
        msa_has_deletion_rows=msa_has_deletion_rows,
        msa_deletion_value_rows=msa_deletion_value_rows,
        max_msa_rows=msa_max_rows,
        affinity_token_mask=affinity_token_mask,
        affinity_mw=affinity_mw,
        template_paths=template_paths,
        max_templates=template_max_count,
        template_include_chains=template_chains,
        bonds=parsed.bonds,
        augment_ref_pos=true,
        batch=1,
        token_atom_names_override=parsed.token_atom_names,
        token_atom_coords_override=parsed.token_atom_coords,
        token_atom_ref_coords_override=parsed.token_atom_ref_coords,
    )

    feats_masked = BoltzGen.boltz_masker(feats; mask=true, mask_backbone=false)
    println("[progress] sampling: starting diffusion (steps=", steps, ", recycles=", recycles, ")")

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

    atom_to_token = feats_masked["atom_to_token"][:, :, 1]
    atom_pad_mask = feats_masked["atom_pad_mask"][:, 1]
    design_m = feats["design_mask"][:, 1]
    for m in 1:size(atom_to_token, 1)
        atom_pad_mask[m] > 0.5 || continue
        t = argmax(view(atom_to_token, m, :))
        if design_m[t] == 0
            feats_out["ref_atom_name_chars"][:, :, m, 1] .= feats["ref_atom_name_chars"][:, :, m, 1]
            feats_out["ref_element"][:, m, 1] .= feats["ref_element"][:, m, 1]
        end
    end
    for t in 1:size(feats["res_type"], 2)
        if design_m[t] == 0
            feats_out["res_type"][:, t, 1] .= feats["res_type"][:, t, 1]
        end
    end

    mkpath(dirname(out_pdb))
    mkpath(dirname(out_pdb37))
    mkpath(dirname(out_cif))
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

    println("Wrote PDB (atom14-like): ", out_pdb)
    println("Wrote PDB (atom37 mapped): ", out_pdb37)
    println("Wrote mmCIF (atom37 mapped): ", out_cif)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
