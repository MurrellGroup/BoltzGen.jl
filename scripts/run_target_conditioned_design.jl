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

function parse_chain_list(spec::AbstractString)
    isempty(strip(spec)) && return nothing
    return [String(strip(s)) for s in split(spec, ',') if !isempty(strip(s))]
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

function main()
    args = parse_kv_args(ARGS)
    target_path = get(args, "target", "")
    isempty(target_path) && error("Missing --target <path-to-pdb-or-cif>")

    include_chains = parse_chain_list(get(args, "target-chains", ""))
    parsed = BoltzGen.load_structure_tokens(target_path; include_chains=include_chains, include_nonpolymer=false)

    d_tokens, d_chain_type = design_tokens(args)
    d_mol_type = BoltzGen.chain_type_ids[d_chain_type]

    residues = copy(parsed.residue_tokens)
    mol_types = copy(parsed.mol_types)
    asym_ids = copy(parsed.asym_ids)
    entity_ids = copy(parsed.entity_ids)
    sym_ids = copy(parsed.sym_ids)
    residue_indices = copy(parsed.residue_indices)
    token_atom_names_override = copy(parsed.token_atom_names)
    token_atom_coords_override = copy(parsed.token_atom_coords)

    T_target = length(residues)
    target_redesign = parse_index_set(get(args, "target-design-mask", ""), T_target)

    if !isempty(d_tokens)
        new_asym = isempty(asym_ids) ? 0 : (maximum(asym_ids) + 1)
        next_res = isempty(residue_indices) ? 1 : (maximum(residue_indices) + 1)
        for (k, tok) in enumerate(d_tokens)
            push!(residues, tok)
            push!(mol_types, d_mol_type)
            push!(asym_ids, new_asym)
            push!(entity_ids, new_asym)
            push!(sym_ids, 0)
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
    weights_path = get(args, "weights", joinpath(WORKSPACE_ROOT, "boltzgen_cache", "boltzgen1_diverse_state_dict.safetensors"))
    steps = parse(Int, get(args, "steps", "100"))
    recycles = parse(Int, get(args, "recycles", "3"))

    out_prefix_default = begin
        base = splitext(basename(target_path))[1]
        joinpath(WORKSPACE_ROOT, "boltzgen_cache", "generated_target_conditioned_" * base)
    end
    out_pdb = get(args, "out-pdb", out_prefix_default * ".pdb")
    out_pdb37 = get(args, "out-pdb-atom37", out_prefix_default * "_atom37.pdb")
    out_cif = get(args, "out-cif", out_prefix_default * ".cif")

    model, _, missing = BoltzGen.load_model_from_safetensors(weights_path)
    if !isempty(missing)
        println("Unmapped state keys: ", length(missing))
    end

    feats = BoltzGen.build_design_features(
        residues;
        mol_types=mol_types,
        asym_ids=asym_ids,
        entity_ids=entity_ids,
        sym_ids=sym_ids,
        residue_indices=residue_indices,
        design_mask=design_mask,
        structure_group=structure_group,
        batch=1,
        token_atom_names_override=token_atom_names_override,
        token_atom_coords_override=token_atom_coords_override,
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

    println("Wrote PDB (atom14-like): ", out_pdb)
    println("Wrote PDB (atom37 mapped): ", out_pdb37)
    println("Wrote mmCIF (atom37 mapped): ", out_cif)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
