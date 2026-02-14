include(normpath(joinpath(@__DIR__, "_activate_runfromhere.jl")))

using CUDA
using cuDNN
using Onion
using SafeTensors
using Random

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

function parse_chain_list(spec::AbstractString)
    isempty(strip(spec)) && return nothing
    return [String(strip(s)) for s in split(spec, ',') if !isempty(strip(s))]
end

function parse_chain_set(spec::AbstractString)
    isempty(strip(spec)) && return nothing
    return Set(String(strip(s)) for s in split(spec, ',') if !isempty(strip(s)))
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

function design_tokens(args::Dict{String,String})
    chain_type = uppercase(get(args, "design-chain-type", "PROTEIN"))
    seq = get(args, "design-sequence", "")
    n = parse(Int, get(args, "design-length", "0"))
    if !isempty(seq)
        return BoltzGen.tokens_from_sequence(seq; chain_type=chain_type), chain_type
    elseif n > 0
        if chain_type == "PROTEIN"
            return fill("GLY", n), chain_type
        elseif chain_type == "DNA"
            return fill("DN", n), chain_type
        elseif chain_type == "RNA"
            return fill("N", n), chain_type
        end
    end
    return String[], chain_type
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
    seed = haskey(args, "seed") ? parse(Int, args["seed"]) : 1
    Random.seed!(seed)
    println("Using seed: ", seed)
    model_family = parse_model_family(get(args, "model-family", "boltzgen1"))
    with_affinity = get(args, "with-affinity", "false") == "true"
    with_confidence_default = model_family == "boltz2" ? "true" : "false"
    with_confidence = get(args, "with-confidence", with_confidence_default) == "true"
    out_heads = get(args, "out-heads", "")
    target_path = get(args, "target", "")
    isempty(target_path) && error("Missing --target <path-to-pdb-or-cif>")

    include_chains = parse_chain_list(get(args, "target-chains", ""))
    target_use_assembly = lowercase(strip(get(args, "target-use-assembly", "false"))) in ("1", "true", "t", "yes", "y", "on")
    include_nonpolymer_default = with_affinity ? "true" : "false"
    include_nonpolymer = get(args, "include-nonpolymer", include_nonpolymer_default) == "true"
    parsed = BoltzGen.load_structure_tokens(
        target_path;
        include_chains=include_chains,
        include_nonpolymer=include_nonpolymer,
        use_assembly=target_use_assembly,
    )

    d_tokens, d_chain_type = design_tokens(args)
    d_mol_type = BoltzGen.chain_type_ids[d_chain_type]

    residues = copy(parsed.residue_tokens)
    mol_types = copy(parsed.mol_types)
    asym_ids = copy(parsed.asym_ids)
    entity_ids = copy(parsed.entity_ids)
    sym_ids = copy(parsed.sym_ids)
    cyclic_period = zeros(Int, length(residues))
    chain_labels = copy(parsed.chain_labels)
    residue_indices = copy(parsed.residue_indices)
    token_atom_names_override = copy(parsed.token_atom_names)
    token_atom_coords_override = copy(parsed.token_atom_coords)

    T_target = length(residues)
    println(
        "[progress] case-ready: target loaded (target_tokens=",
        T_target,
        ", added_design_tokens=",
        length(d_tokens),
        ", target=",
        target_path,
        ")",
    )
    target_redesign = parse_index_set(get(args, "target-design-mask", ""), T_target)

    if !isempty(d_tokens)
        new_asym = isempty(asym_ids) ? 0 : (maximum(asym_ids) + 1)
        next_res = isempty(residue_indices) ? 1 : (maximum(residue_indices) + 1)
        design_cyclic = lowercase(strip(get(args, "design-cyclic", "false"))) in ("1", "true", "t", "yes", "y", "on")
        design_cyclic_period = design_cyclic ? length(d_tokens) : 0
        for (k, tok) in enumerate(d_tokens)
            push!(residues, tok)
            push!(mol_types, d_mol_type)
            push!(asym_ids, new_asym)
            push!(entity_ids, new_asym)
            push!(sym_ids, 0)
            push!(cyclic_period, design_cyclic_period)
            push!(chain_labels, "DESIGN")
            push!(residue_indices, next_res + k - 1)
            push!(token_atom_names_override, String[])
            push!(token_atom_coords_override, Dict{String,NTuple{3,Float32}}())
        end
    end

    T_total = length(residues)
    design_mask = falses(T_total)
    for i in 1:T_target
        design_mask[i] = target_redesign[i]
    end
    for i in (T_target + 1):T_total
        design_mask[i] = true
    end

    structure_group = vcat(fill(1, T_target), fill(0, T_total - T_target))
    affinity_chain_set = haskey(args, "affinity-chains") ? parse_chain_set(args["affinity-chains"]) : nothing
    affinity_token_mask = if haskey(args, "affinity-mask")
        parse_index_set(args["affinity-mask"], T_total)
    elseif affinity_chain_set !== nothing
        [chain_labels[i] in affinity_chain_set for i in 1:T_total]
    elseif with_affinity
        [mol_types[i] == BoltzGen.chain_type_ids["NONPOLYMER"] for i in 1:T_total]
    else
        falses(T_total)
    end
    affinity_mw = haskey(args, "affinity-mw") ? Float32(parse(Float64, args["affinity-mw"])) : 0f0

    if with_affinity && !any(affinity_token_mask)
        println("Warning: affinity enabled but affinity_token_mask is empty.")
    end

    weights_spec = get(args, "weights", default_weights_for_family(model_family, with_affinity))
    if model_family == "boltz2" && any(design_mask)
        error("Boltz2 mode is folding-only in this script; redesign masks/added design chains are not supported.")
    end
    weights_path = require_sampling_checkpoint!(weights_spec; requires_design_conditioning=any(design_mask))
    steps = parse(Int, get(args, "steps", "100"))
    recycles = parse(Int, get(args, "recycles", "3"))
    msa_file = get(args, "msa-file", "")
    msa_max_rows = haskey(args, "msa-max-rows") ? parse(Int, args["msa-max-rows"]) : nothing
    msa_sequences = isempty(msa_file) ? nothing : BoltzGen.load_msa_sequences(msa_file; max_rows=msa_max_rows)
    template_paths = parse_string_list(get(args, "template-paths", ""))
    template_max_count = haskey(args, "template-max-count") ? parse(Int, args["template-max-count"]) : nothing
    template_chains = haskey(args, "template-chains") ? parse_chain_list(args["template-chains"]) : nothing

    out_prefix_default = begin
        base = splitext(basename(target_path))[1]
        joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_target_conditioned_" * base)
    end
    out_pdb = get(args, "out-pdb", out_prefix_default * ".pdb")
    out_pdb37 = get(args, "out-pdb-atom37", out_prefix_default * "_atom37.pdb")
    out_cif = get(args, "out-cif", out_prefix_default * ".cif")

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
        residues;
        mol_types=mol_types,
        asym_ids=asym_ids,
        entity_ids=entity_ids,
        sym_ids=sym_ids,
        cyclic_period=cyclic_period,
        residue_indices=residue_indices,
        design_mask=design_mask,
        structure_group=structure_group,
        msa_sequences=msa_sequences,
        max_msa_rows=msa_max_rows,
        affinity_token_mask=affinity_token_mask,
        affinity_mw=affinity_mw,
        template_paths=template_paths,
        max_templates=template_max_count,
        template_include_chains=template_chains,
        batch=1,
        token_atom_names_override=token_atom_names_override,
        token_atom_coords_override=token_atom_coords_override,
    )

    feats_masked = BoltzGen.boltz_masker(feats; mask=true, mask_backbone=false)

    is_design = any(design_mask)
    override_step_scale = (is_design && model_family == "boltzgen1") ? 1.8f0 : nothing
    override_noise_scale = (is_design && model_family == "boltzgen1") ? 0.95f0 : nothing
    println("[progress] sampling: starting diffusion (steps=", steps, ", recycles=", recycles, ")")
    out = BoltzGen.boltz_forward(
        model,
        feats_masked;
        recycling_steps=recycles,
        num_sampling_steps=steps,
        diffusion_samples=1,
        step_scale=override_step_scale,
        noise_scale=override_noise_scale,
        inference_logging=false,
    )

    coords = out["sample_atom_coords"]
    feats_out = BoltzGen.postprocess_atom14(feats_masked, coords)

    # Keep fixed target atoms and residue identities unchanged.
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

    println("Wrote PDB (atom14-like): ", out_pdb)
    println("Wrote PDB (atom37 mapped): ", out_pdb37)
    println("Wrote mmCIF (atom37 mapped): ", out_cif)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
