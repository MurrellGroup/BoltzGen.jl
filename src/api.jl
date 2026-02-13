"""
    REPL-friendly API for BoltzGen.jl

Provides `BoltzGenHandle` for holding a loaded model, `load_boltzgen` / `load_boltz2`
for one-time model loading, and high-level functions that map to the existing scripts
but return results as dictionaries for interactive use.
"""

using SafeTensors

# ── Handle ──────────────────────────────────────────────────────────────────────

struct BoltzGenHandle
    model::BoltzModel
    config::Dict
    weights_name::String
    model_family::String       # "boltzgen1" or "boltz2"
    has_confidence::Bool
    has_affinity::Bool
    on_gpu::Bool
end

function Base.show(io::IO, h::BoltzGenHandle)
    print(io, "BoltzGenHandle(", h.model_family, ", weights=\"", h.weights_name, "\"",
        h.has_confidence ? ", confidence" : "",
        h.has_affinity ? ", affinity" : "",
        h.on_gpu ? ", gpu" : "", ")")
end

# ── Validation helpers ──────────────────────────────────────────────────────────

function _validate_design_handle(handle::BoltzGenHandle)
    handle.model_family == "boltzgen1" || error(
        "Design functions require a boltzgen1 model (got model_family=\"$(handle.model_family)\"). Use load_boltzgen().",
    )
end

function _validate_fold_handle(handle::BoltzGenHandle)
    handle.has_confidence || error(
        "Fold functions require a Boltz2 model with confidence (got model_family=\"$(handle.model_family)\"). Use load_boltz2().",
    )
end

# ── Checkpoint validation (matches script pattern) ──────────────────────────────

function _load_and_validate_checkpoint(weights_spec::AbstractString; requires_design_conditioning::Bool=false)
    weights_path = resolve_weights_path(weights_spec)
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
    return state, weights_path
end

# ── GPU/device helpers ──────────────────────────────────────────────────────────

"""Get a model parameter array to use as device reference for array allocation."""
function _device_ref(handle::BoltzGenHandle)
    handle.model.s_init.weight
end

"""Transfer feature dict arrays to the model's device (GPU if on_gpu, else no-op)."""
function _feats_to_device(feats::Dict, handle::BoltzGenHandle)
    handle.on_gpu || return feats
    ref = _device_ref(handle)
    gpu_feats = Dict{String,Any}()
    for (k, v) in feats
        if v isa AbstractArray
            gpu_feats[k] = copyto!(similar(ref, eltype(v), size(v)), v)
        else
            gpu_feats[k] = v
        end
    end
    return gpu_feats
end

"""Convert all arrays in a dict to CPU Arrays."""
function _dict_to_cpu(d::Dict)
    Dict{String,Any}(k => (v isa AbstractArray ? Array(v) : v) for (k, v) in d)
end

# ── Load functions ──────────────────────────────────────────────────────────────

"""
    load_boltzgen(; weights="boltzgen1_diverse_state_dict.safetensors", gpu=false) → BoltzGenHandle

Load a BoltzGen1 design model. Call once, reuse across multiple generations.
Pass `gpu=true` to move the model to GPU.
"""
function load_boltzgen(; weights::AbstractString="boltzgen1_diverse_state_dict.safetensors", gpu::Bool=false)
    state, weights_path = _load_and_validate_checkpoint(weights; requires_design_conditioning=true)
    model, cfg, missing_keys = load_model_from_state(
        state;
        confidence_prediction=false,
        affinity_prediction=false,
    )
    if !isempty(missing_keys)
        println("Unmapped state keys: ", length(missing_keys))
    end
    if gpu
        model = Onion.Flux.gpu(model)
    end
    return BoltzGenHandle(model, cfg, basename(weights_path), "boltzgen1", false, false, gpu)
end

"""
    load_boltz2(; affinity=false, weights=nothing, gpu=false) → BoltzGenHandle

Load a Boltz2 folding model. Uses boltz2_conf by default, or boltz2_aff if `affinity=true`.
Pass `gpu=true` to move the model to GPU.
"""
function load_boltz2(; affinity::Bool=false, weights::Union{Nothing,AbstractString}=nothing, gpu::Bool=false)
    w = if weights !== nothing
        weights
    else
        default_weights_filename("boltz2", affinity)
    end
    state, weights_path = _load_and_validate_checkpoint(w; requires_design_conditioning=false)
    model, cfg, missing_keys = load_model_from_state(
        state;
        confidence_prediction=true,
        affinity_prediction=affinity,
    )
    if !isempty(missing_keys)
        println("Unmapped state keys: ", length(missing_keys))
    end
    if gpu
        model = Onion.Flux.gpu(model)
    end
    return BoltzGenHandle(model, cfg, basename(weights_path), "boltz2", true, affinity, gpu)
end

# ── _restore_fixed_tokens! ──────────────────────────────────────────────────────

"""
Restore original (pre-mask) atom names/elements for non-designed or non-protein tokens.
This is the common post-processing pattern extracted from all design/fold scripts.
Operates on CPU arrays only (uses loops and argmax).
"""
function _restore_fixed_tokens!(feats_out::Dict, feats_orig::Dict; batch::Int=1)
    atom_to_token = feats_out["atom_to_token"][:, :, batch]
    atom_pad_mask = feats_out["atom_pad_mask"][:, batch]
    design_m = feats_orig["design_mask"][:, batch]
    mol_type = feats_orig["mol_type"][:, batch]
    protein_id = chain_type_ids["PROTEIN"]
    for m in axes(atom_to_token, 1)
        atom_pad_mask[m] > 0.5 || continue
        t = argmax(view(atom_to_token, m, :))
        is_designed_protein = (design_m[t] > 0) && (mol_type[t] == protein_id)
        if !is_designed_protein
            feats_out["ref_atom_name_chars"][:, :, m, batch] .= feats_orig["ref_atom_name_chars"][:, :, m, batch]
            feats_out["ref_element"][:, m, batch] .= feats_orig["ref_element"][:, m, batch]
        end
    end
    for t in axes(feats_orig["res_type"], 2)
        is_designed_protein = (design_m[t] > 0) && (mol_type[t] == protein_id)
        if !is_designed_protein
            feats_out["res_type"][:, t, batch] .= feats_orig["res_type"][:, t, batch]
        end
    end
    return feats_out
end

# ── Common forward + postprocess ────────────────────────────────────────────────

function _run_forward(handle::BoltzGenHandle, feats::Dict, feats_masked::Dict;
    steps::Int, recycles::Int, batch::Int=1)

    # Transfer masked features to model device (GPU if on_gpu)
    feats_fwd = _feats_to_device(feats_masked, handle)

    out = boltz_forward(
        handle.model,
        feats_fwd;
        recycling_steps=recycles,
        num_sampling_steps=steps,
        diffusion_samples=1,
        inference_logging=false,
    )

    # Post-processing requires CPU arrays (loops, argmax, norm in postprocess_atom14)
    coords = Array(out["sample_atom_coords"])
    feats_fwd_cpu = _dict_to_cpu(feats_fwd)

    feats_out = postprocess_atom14(feats_fwd_cpu, coords)
    _restore_fixed_tokens!(feats_out, feats; batch=batch)

    # Build result with all arrays on CPU
    result = _dict_to_cpu(out)
    result["feats"] = feats_out
    result["feats_orig"] = feats
    result["coords"] = coords
    return result
end

# ── design_from_sequence ────────────────────────────────────────────────────────

"""
    design_from_sequence(handle, sequence; length=0, chain_type="PROTEIN", ...) → Dict

Maps to `run_design_from_sequence.jl`. Provide either a non-empty `sequence` or `length > 0`
for de novo design.
"""
function design_from_sequence(
    handle::BoltzGenHandle,
    sequence::AbstractString="";
    length::Int=0,
    chain_type::String="PROTEIN",
    design_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    steps::Int=100,
    recycles::Int=3,
    seed::Union{Nothing,Int}=nothing,
    cyclic::Bool=false,
)
    _validate_design_handle(handle)
    seed !== nothing && Random.seed!(seed)

    residues = if !isempty(sequence)
        tokens_from_sequence(sequence; chain_type=chain_type)
    elseif length > 0
        if chain_type == "PROTEIN"
            fill("GLY", length)
        elseif chain_type == "DNA"
            fill("DN", length)
        elseif chain_type == "RNA"
            fill("N", length)
        else
            error("Unsupported chain-type: $chain_type")
        end
    else
        error("Provide either a non-empty sequence or length > 0")
    end

    T = Base.length(residues)
    dm = if design_mask !== nothing
        design_mask
    else
        isempty(sequence) ? trues(T) : falses(T)
    end

    mol_type_id = chain_type_ids[chain_type]
    mol_types = fill(mol_type_id, T)
    cyclic_period = cyclic ? fill(T, T) : zeros(Int, T)

    feats = build_design_features(
        residues;
        mol_types=mol_types,
        cyclic_period=cyclic_period,
        design_mask=dm,
        batch=1,
    )

    feats_masked = boltz_masker(feats; mask=true, mask_backbone=false)
    return _run_forward(handle, feats, feats_masked; steps=steps, recycles=recycles, batch=1)
end

# ── design_from_yaml ────────────────────────────────────────────────────────────

"""
    design_from_yaml(handle, yaml_path; steps=100, recycles=3, seed=nothing, ...) → Dict

Maps to `run_design_from_yaml.jl`.
"""
function design_from_yaml(
    handle::BoltzGenHandle,
    yaml_path::AbstractString;
    steps::Int=100,
    recycles::Int=3,
    seed::Union{Nothing,Int}=nothing,
    include_nonpolymer::Bool=true,
)
    _validate_design_handle(handle)
    seed !== nothing && Random.seed!(seed)

    parsed = parse_design_yaml(yaml_path; include_nonpolymer=include_nonpolymer)

    msa_sequences = parsed.msa_sequences
    msa_paired_rows = get(parsed, :msa_paired_rows, nothing)
    msa_has_deletion_rows = get(parsed, :msa_has_deletion_rows, nothing)
    msa_deletion_value_rows = get(parsed, :msa_deletion_value_rows, nothing)

    feats = build_design_features(
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
        bonds=parsed.bonds,
        augment_ref_pos=true,
        batch=1,
        token_atom_names_override=parsed.token_atom_names,
        token_atom_coords_override=parsed.token_atom_coords,
        token_atom_ref_coords_override=parsed.token_atom_ref_coords,
    )

    feats_masked = boltz_masker(feats; mask=true, mask_backbone=false)
    return _run_forward(handle, feats, feats_masked; steps=steps, recycles=recycles, batch=1)
end

# ── fold_from_sequence ──────────────────────────────────────────────────────────

"""
    fold_from_sequence(handle, sequence; steps=100, recycles=3, seed=nothing) → Dict

Maps to `run_fold_from_sequence.jl`. Requires a Boltz2 handle (from `load_boltz2()`).
"""
function fold_from_sequence(
    handle::BoltzGenHandle,
    sequence::AbstractString;
    chain_type::String="PROTEIN",
    steps::Int=100,
    recycles::Int=3,
    seed::Union{Nothing,Int}=nothing,
)
    _validate_fold_handle(handle)
    seed !== nothing && Random.seed!(seed)

    residues = tokens_from_sequence(sequence; chain_type=chain_type)
    T = Base.length(residues)

    mol_type_id = chain_type_ids[chain_type]
    mol_types = fill(mol_type_id, T)
    cyclic_period = zeros(Int, T)
    dm = falses(T)

    feats = build_design_features(
        residues;
        mol_types=mol_types,
        cyclic_period=cyclic_period,
        design_mask=dm,
        batch=1,
    )

    feats_masked = boltz_masker(feats; mask=true, mask_backbone=false)
    return _run_forward(handle, feats, feats_masked; steps=steps, recycles=recycles, batch=1)
end

# ── fold_from_structure ─────────────────────────────────────────────────────────

"""
    fold_from_structure(handle, target_path; steps=100, recycles=3, seed=nothing, ...) → Dict

Maps to `run_fold_from_structure.jl`. Requires a Boltz2 handle.
"""
function fold_from_structure(
    handle::BoltzGenHandle,
    target_path::AbstractString;
    include_chains::Union{Nothing,Vector{String}}=nothing,
    include_nonpolymer::Bool=true,
    steps::Int=100,
    recycles::Int=3,
    seed::Union{Nothing,Int}=nothing,
)
    _validate_fold_handle(handle)
    seed !== nothing && Random.seed!(seed)

    parsed = load_structure_tokens(
        target_path;
        include_chains=include_chains,
        include_nonpolymer=include_nonpolymer,
    )

    T = Base.length(parsed.residue_tokens)
    design_mask = falses(T)
    structure_group = fill(1, T)

    affinity_token_mask = if handle.has_affinity
        [parsed.mol_types[i] == chain_type_ids["NONPOLYMER"] for i in 1:T]
    else
        falses(T)
    end

    feats = build_design_features(
        parsed.residue_tokens;
        mol_types=parsed.mol_types,
        asym_ids=parsed.asym_ids,
        entity_ids=parsed.entity_ids,
        sym_ids=parsed.sym_ids,
        cyclic_period=zeros(Int, T),
        residue_indices=parsed.residue_indices,
        design_mask=design_mask,
        structure_group=structure_group,
        affinity_token_mask=affinity_token_mask,
        batch=1,
        token_atom_names_override=parsed.token_atom_names,
        token_atom_coords_override=parsed.token_atom_coords,
    )

    feats_masked = boltz_masker(feats; mask=true, mask_backbone=false)
    return _run_forward(handle, feats, feats_masked; steps=steps, recycles=recycles, batch=1)
end

# ── target_conditioned_design ───────────────────────────────────────────────────

"""
    target_conditioned_design(handle, target_path; design_length=8, ...) → Dict

Maps to `run_target_conditioned_design.jl`.
"""
function target_conditioned_design(
    handle::BoltzGenHandle,
    target_path::AbstractString;
    design_length::Int=8,
    design_chain_type::String="PROTEIN",
    include_chains::Union{Nothing,Vector{String}}=nothing,
    include_nonpolymer::Bool=false,
    steps::Int=100,
    recycles::Int=3,
    seed::Union{Nothing,Int}=nothing,
)
    _validate_design_handle(handle)
    seed !== nothing && Random.seed!(seed)

    parsed = load_structure_tokens(
        target_path;
        include_chains=include_chains,
        include_nonpolymer=include_nonpolymer,
    )

    residues = copy(parsed.residue_tokens)
    mol_types = copy(parsed.mol_types)
    asym_ids = copy(parsed.asym_ids)
    entity_ids = copy(parsed.entity_ids)
    sym_ids = copy(parsed.sym_ids)
    cyclic_period = zeros(Int, Base.length(residues))
    residue_indices = copy(parsed.residue_indices)
    token_atom_names_override = copy(parsed.token_atom_names)
    token_atom_coords_override = copy(parsed.token_atom_coords)

    T_target = Base.length(residues)

    if design_length > 0
        d_mol_type = chain_type_ids[design_chain_type]
        d_tokens = if design_chain_type == "PROTEIN"
            fill("GLY", design_length)
        elseif design_chain_type == "DNA"
            fill("DN", design_length)
        elseif design_chain_type == "RNA"
            fill("N", design_length)
        else
            error("Unsupported design-chain-type: $design_chain_type")
        end

        new_asym = isempty(asym_ids) ? 0 : (maximum(asym_ids) + 1)
        next_res = isempty(residue_indices) ? 1 : (maximum(residue_indices) + 1)
        for (k, tok) in enumerate(d_tokens)
            push!(residues, tok)
            push!(mol_types, d_mol_type)
            push!(asym_ids, new_asym)
            push!(entity_ids, new_asym)
            push!(sym_ids, 0)
            push!(cyclic_period, 0)
            push!(residue_indices, next_res + k - 1)
            push!(token_atom_names_override, String[])
            push!(token_atom_coords_override, Dict{String,NTuple{3,Float32}}())
        end
    end

    T_total = Base.length(residues)
    dm = falses(T_total)
    for i in (T_target + 1):T_total
        dm[i] = true
    end

    structure_group = vcat(fill(1, T_target), fill(0, T_total - T_target))

    feats = build_design_features(
        residues;
        mol_types=mol_types,
        asym_ids=asym_ids,
        entity_ids=entity_ids,
        sym_ids=sym_ids,
        cyclic_period=cyclic_period,
        residue_indices=residue_indices,
        design_mask=dm,
        structure_group=structure_group,
        batch=1,
        token_atom_names_override=token_atom_names_override,
        token_atom_coords_override=token_atom_coords_override,
    )

    feats_masked = boltz_masker(feats; mask=true, mask_backbone=false)
    return _run_forward(handle, feats, feats_masked; steps=steps, recycles=recycles, batch=1)
end

# ── denovo_sample ───────────────────────────────────────────────────────────────

"""
    denovo_sample(handle, length; steps=100, recycles=3, seed=nothing) → Dict

Maps to `run_denovo_sample.jl`. Uses `build_denovo_atom14_features` (simplest path).
"""
function denovo_sample(
    handle::BoltzGenHandle,
    length::Int;
    steps::Int=100,
    recycles::Int=3,
    seed::Union{Nothing,Int}=nothing,
)
    _validate_design_handle(handle)
    seed !== nothing && Random.seed!(seed)

    feats = build_denovo_atom14_features(length)
    feats_masked = boltz_masker(feats; mask=true, mask_backbone=false)
    return _run_forward(handle, feats, feats_masked; steps=steps, recycles=recycles, batch=1)
end

# ── Output functions ────────────────────────────────────────────────────────────

"""
    output_to_pdb(result; batch=1) → String

Generate PDB (atom14-like) string from a result dict.
"""
function output_to_pdb(result::Dict; batch::Int=1)
    io = IOBuffer()
    write_pdb(io, result["feats"], result["coords"]; batch=batch)
    return String(take!(io))
end

"""
    output_to_pdb_atom37(result; batch=1) → String

Generate PDB (atom37 mapped) string from a result dict.
"""
function output_to_pdb_atom37(result::Dict; batch::Int=1)
    io = IOBuffer()
    write_pdb_atom37(io, result["feats"], result["coords"]; batch=batch)
    return String(take!(io))
end

"""
    output_to_mmcif(result; batch=1) → String

Generate mmCIF string from a result dict.
"""
function output_to_mmcif(result::Dict; batch::Int=1)
    io = IOBuffer()
    write_mmcif(io, result["feats"], result["coords"]; batch=batch)
    return String(take!(io))
end

"""
    write_outputs(result, prefix; batch=1)

Write `prefix_atom14.pdb`, `prefix_atom37.pdb`, and `prefix_atom37.cif` files.
"""
function write_outputs(result::Dict, prefix::AbstractString; batch::Int=1)
    write_pdb(prefix * "_atom14.pdb", result["feats"], result["coords"]; batch=batch)
    write_pdb_atom37(prefix * "_atom37.pdb", result["feats"], result["coords"]; batch=batch)
    write_mmcif(prefix * "_atom37.cif", result["feats"], result["coords"]; batch=batch)
end

"""
    confidence_metrics(result) → NamedTuple

Extract confidence/affinity head values from a result dict.
Returns a NamedTuple with keys that are present in the result.
"""
function confidence_metrics(result::Dict)
    head_keys = (
        "ptm", "iptm", "complex_plddt", "complex_iplddt",
        "affinity_pred_value", "affinity_probability_binary",
        "affinity_pred_value1", "affinity_probability_binary1",
        "affinity_pred_value2", "affinity_probability_binary2",
    )
    pairs = Pair{Symbol,Any}[]
    for k in head_keys
        if haskey(result, k)
            push!(pairs, Symbol(k) => vec(Float32.(result[k])))
        end
    end
    return (; pairs...)
end
