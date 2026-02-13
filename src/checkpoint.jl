const BOLTZGEN_HF_REPO_ID = "MurrellLab/BoltzGen.jl"
const BOLTZGEN_HF_REPO_TYPE = nothing
const BOLTZGEN_CHECKPOINT_ALIASES = Dict(
    "boltzgen1_diverse" => "boltzgen1_diverse_state_dict.safetensors",
    "boltzgen1_adherence" => "boltzgen1_adherence_state_dict.safetensors",
    "boltz2_conf_final" => "boltz2_conf_final_state_dict.safetensors",
    "boltz2_aff" => "boltz2_aff_state_dict.safetensors",
)

function default_weights_filename(model_family::AbstractString, with_affinity::Bool)
    fam = lowercase(strip(String(model_family)))
    fam in ("boltzgen1", "boltz2") || error(
        "Unsupported model family '$model_family' (expected boltzgen1 or boltz2)",
    )
    if fam == "boltz2"
        return with_affinity ? "boltz2_aff_state_dict.safetensors" : "boltz2_conf_final_state_dict.safetensors"
    end
    return "boltzgen1_diverse_state_dict.safetensors"
end

function _normalize_weights_filename(weights_spec::AbstractString)
    spec = strip(String(weights_spec))
    isempty(spec) && error("Empty weights spec provided.")
    if startswith(spec, "/") || startswith(spec, ".") || startswith(spec, "~") || occursin('\\', spec)
        error(
            "Local weights paths are disabled. Use a HuggingFace alias or safetensors filename from " *
            "$(BOLTZGEN_HF_REPO_ID).",
        )
    end
    haskey(BOLTZGEN_CHECKPOINT_ALIASES, spec) && return BOLTZGEN_CHECKPOINT_ALIASES[spec]
    endswith(lowercase(spec), ".safetensors") && return spec
    error(
        "Unsupported weights spec '$weights_spec'. " *
        "Use a safetensors filename or a supported alias.",
    )
end

function resolve_weights_path(
    weights_spec::AbstractString;
    repo_id::AbstractString=BOLTZGEN_HF_REPO_ID,
    revision::Union{Nothing,AbstractString}=nothing,
)
    spec = strip(String(weights_spec))
    isempty(spec) && error("Empty weights spec provided.")

    # Local path loading intentionally disabled.
    # Previous local-path logic:
    # if isfile(spec)
    #     return abspath(spec)
    # end
    # if startswith(spec, "/") || startswith(spec, ".") || startswith(spec, "~")
    #     expanded = startswith(spec, "~") ? expanduser(spec) : spec
    #     resolved = abspath(expanded)
    #     isfile(resolved) || error("Weights file not found at explicit path: $resolved")
    #     return resolved
    # end

    filename = _normalize_weights_filename(spec)
    return isnothing(revision) ?
        hf_hub_download(repo_id, filename; repo_type=BOLTZGEN_HF_REPO_TYPE) :
        hf_hub_download(repo_id, filename; repo_type=BOLTZGEN_HF_REPO_TYPE, revision=revision)
end

function normalize_state_key(key::String)
    key = replace(key, ".proj_z.0." => ".proj_z_norm.")
    key = replace(key, ".proj_z.1." => ".proj_z.")
    key = replace(key, ".output_projection.0." => ".output_projection.")
    key = replace(key, ".output_projection_linear." => ".output_projection.")
    key = replace(key, ".fourier_embedding.proj." => ".fourier_embedding.")
    key = replace(key, ".fourier_embed.proj." => ".fourier_embed.")
    key = replace(key, ".token_transformer_layers.0." => ".token_transformer.")
    return key
end

function _get_child(obj, token::AbstractString)
    if occursin(r"^\d+$", token)
        idx = parse(Int, token) + 1
        if obj isa Tuple
            return obj[idx]
        elseif obj isa AbstractVector
            return obj[idx]
        else
            return obj
        end
    end
    return getfield(obj, Symbol(token))
end

function _assign_param!(model, key::String, value)
    nk = normalize_state_key(key)
    parts = split(nk, ".")
    obj = model
    for i in 1:length(parts)-1
        obj = try
            _get_child(obj, parts[i])
        catch
            return false
        end
    end
    pname = parts[end]
    target = if hasfield(typeof(obj), Symbol(pname))
        getfield(obj, Symbol(pname))
    elseif pname == "weight" && hasfield(typeof(obj), :w)
        getfield(obj, :w)
    elseif pname == "bias" && hasfield(typeof(obj), :b)
        getfield(obj, :b)
    else
        return false
    end

    val = Float32.(value)
    if obj isa BGEmbedding && pname == "weight"
        if size(target) == size(val)
            target .= val
        elseif ndims(val) == 2 && size(target) == (size(val, 2), size(val, 1))
            target .= permutedims(val, (2, 1))
        else
            error("shape mismatch for $nk: target=$(size(target)) val=$(size(val))")
        end
        return true
    end

    if size(target) != size(val)
        error("shape mismatch for $nk: target=$(size(target)) val=$(size(val))")
    end
    target .= val
    return true
end

function load_params!(model, state::AbstractDict{String, <:AbstractArray})
    missing = String[]
    for (k, v) in state
        ok = _assign_param!(model, k, v)
        ok || push!(missing, k)
    end
    return missing
end

function _count_layers(keys, prefix::String)
    idx = Set{Int}()
    prefix_dot = prefix * "."
    for k in keys
        startswith(k, prefix_dot) || continue
        rest = k[length(prefix_dot)+1:end]
        parts = split(rest, ".")
        isempty(parts) && continue
        num = try
            parse(Int, parts[1])
        catch
            continue
        end
        push!(idx, num)
    end
    return length(idx)
end

function _has_prefix(keys, prefix::String)
    p = prefix * "."
    return any(startswith(k, p) for k in keys)
end

function _infer_confidence_model_args(state, state_keys, token_z::Int)
    _has_prefix(state_keys, "confidence_module") || return nothing

    conf_blocks = _count_layers(state_keys, "confidence_module.pairformer_stack.layers")
    conf_heads_key = "confidence_module.pairformer_stack.layers.0.attention.proj_z.1.weight"
    conf_pair_key = "confidence_module.pairformer_stack.layers.0.tri_att_start.linear.weight"
    conf_num_heads = haskey(state, conf_heads_key) ? Int(size(state[conf_heads_key], 1)) : 16
    conf_pair_heads = haskey(state, conf_pair_key) ? Int(size(state[conf_pair_key], 1)) : 4

    plddt_key = "confidence_module.confidence_heads.to_plddt_logits.weight"
    pde_key = "confidence_module.confidence_heads.to_pde_logits.weight"
    pde_intra_key = "confidence_module.confidence_heads.to_pde_intra_logits.weight"
    pae_key = "confidence_module.confidence_heads.to_pae_logits.weight"
    pae_intra_key = "confidence_module.confidence_heads.to_pae_intra_logits.weight"
    dist_key = "confidence_module.dist_bin_pairwise_embed.weight"

    token_level_confidence = haskey(state, plddt_key) ? (size(state[plddt_key], 1) <= 64) : true
    num_plddt_bins = if haskey(state, plddt_key)
        token_level_confidence ? Int(size(state[plddt_key], 1)) : 50
    else
        50
    end
    num_pde_bins = haskey(state, pde_key) ? Int(size(state[pde_key], 1)) :
        (haskey(state, pde_intra_key) ? Int(size(state[pde_intra_key], 1)) : 64)
    num_pae_bins = haskey(state, pae_key) ? Int(size(state[pae_key], 1)) :
        (haskey(state, pae_intra_key) ? Int(size(state[pae_intra_key], 1)) : 64)
    use_separate_heads = haskey(state, pde_intra_key) || haskey(state, pae_intra_key) ||
        haskey(state, "confidence_module.confidence_heads.to_pde_inter_logits.weight") ||
        haskey(state, "confidence_module.confidence_heads.to_pae_inter_logits.weight")

    add_s_to_z_prod = any(startswith(k, "confidence_module.s_to_z_prod_in1.") for k in state_keys)
    add_s_input_to_s = any(startswith(k, "confidence_module.s_input_to_s.") for k in state_keys)
    no_update_s = !any(startswith(k, "confidence_module.s_norm.") for k in state_keys)
    add_z_input_to_z = any(startswith(k, "confidence_module.rel_pos.") for k in state_keys)

    return Dict{Symbol,Any}(
        :pairformer_args => Dict{Symbol,Any}(
            :num_blocks => max(conf_blocks, 1),
            :num_heads => conf_num_heads,
            :pairwise_head_width => Int(token_z ÷ max(conf_pair_heads, 1)),
            :pairwise_num_heads => conf_pair_heads,
            :dropout => 0.0,
            :post_layer_norm => false,
            :activation_checkpointing => false,
        ),
        :num_dist_bins => haskey(state, dist_key) ? Int(size(state[dist_key], 1)) : 64,
        :token_level_confidence => token_level_confidence,
        :max_dist => 22,
        :add_s_to_z_prod => add_s_to_z_prod,
        :add_s_input_to_s => add_s_input_to_s,
        :no_update_s => no_update_s,
        :add_z_input_to_z => add_z_input_to_z,
        :confidence_args => Dict{Symbol,Any}(
            :num_plddt_bins => num_plddt_bins,
            :num_pde_bins => num_pde_bins,
            :num_pae_bins => num_pae_bins,
            :use_separate_heads => use_separate_heads,
        ),
    )
end

function _infer_affinity_model_args(state, state_keys, prefix::String, token_s::Int, token_z::Int)
    _has_prefix(state_keys, prefix) || return nothing

    pair_prefix = prefix * ".pairformer_stack.layers"
    aff_blocks = _count_layers(state_keys, pair_prefix)
    aff_pair_key = prefix * ".pairformer_stack.layers.0.tri_att_start.linear.weight"
    aff_pair_heads = haskey(state, aff_pair_key) ? Int(size(state[aff_pair_key], 1)) : 4
    dist_key = prefix * ".dist_bin_pairwise_embed.weight"
    aff_token_s_key = prefix * ".affinity_heads.to_affinity_pred_value.0.weight"
    aff_token_s = haskey(state, aff_token_s_key) ? Int(size(state[aff_token_s_key], 2)) : token_s

    return Dict{Symbol,Any}(
        :pairformer_args => Dict{Symbol,Any}(
            :num_blocks => max(aff_blocks, 1),
            :pairwise_head_width => Int(token_z ÷ max(aff_pair_heads, 1)),
            :pairwise_num_heads => aff_pair_heads,
            :dropout => 0.0,
            :post_layer_norm => false,
            :activation_checkpointing => false,
        ),
        :transformer_args => Dict{Symbol,Any}(
            :token_s => aff_token_s,
            :num_blocks => 1,
            :num_heads => 1,
            :activation_checkpointing => false,
        ),
        :num_dist_bins => haskey(state, dist_key) ? Int(size(state[dist_key], 1)) : 64,
        :max_dist => 22,
    )
end

function infer_config(state::AbstractDict{String, <:AbstractArray})
    state_keys = collect(keys(state))

    token_s = size(state["input_embedder.res_type_encoding.weight"], 1)
    token_z = size(state["z_init_1.weight"], 1)
    atom_s = size(state["input_embedder.atom_encoder.embed_atom_features.weight"], 1)
    atom_feature_dim = size(state["input_embedder.atom_encoder.embed_atom_features.weight"], 2)
    atom_z = size(state["input_embedder.atom_encoder.embed_atompair_ref_pos.weight"], 1)
    num_bins = size(state["distogram_module.distogram.weight"], 1)

    atom_encoder_depth_raw = _count_layers(state_keys, "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers")
    atom_encoder_depth = atom_encoder_depth_raw > 0 ? atom_encoder_depth_raw : 3
    atom_encoder_heads = Int(size(state["input_embedder.atom_enc_proj_z.1.weight"], 1) ÷ max(atom_encoder_depth, 1))

    msa_blocks = _count_layers(state_keys, "msa_module.layers")
    pairformer_blocks = _count_layers(state_keys, "pairformer_module.layers")
    template_blocks = _count_layers(state_keys, "template_module.pairformer.layers")
    token_distance_blocks = _count_layers(state_keys, "token_distance_module.pairformer.layers")

    pairformer_heads = size(state["pairformer_module.layers.0.attention.proj_z.1.weight"], 1)
    pairwise_num_heads = size(state["pairformer_module.layers.0.tri_att_start.linear.weight"], 1)
    pairwise_head_width = Int(token_z ÷ pairwise_num_heads)

    atom_enc_heads_diff = Int(size(state["diffusion_conditioning.atom_enc_proj_z.0.1.weight"], 1))
    atom_dec_heads_diff = Int(size(state["diffusion_conditioning.atom_dec_proj_z.0.1.weight"], 1))
    atom_enc_depth_diff_raw = _count_layers(state_keys, "diffusion_conditioning.atom_enc_proj_z")
    atom_dec_depth_diff_raw = _count_layers(state_keys, "diffusion_conditioning.atom_dec_proj_z")
    atom_enc_depth_diff = atom_enc_depth_diff_raw > 0 ? atom_enc_depth_diff_raw : 3
    atom_dec_depth_diff = atom_dec_depth_diff_raw > 0 ? atom_dec_depth_diff_raw : 3

    token_transformer_depth_raw = _count_layers(state_keys, "structure_module.score_model.token_transformer_layers.0.layers")
    token_layers_raw = _count_layers(state_keys, "structure_module.score_model.token_transformer_layers")
    token_transformer_depth = token_transformer_depth_raw > 0 ? token_transformer_depth_raw : 24
    token_layers = token_layers_raw > 0 ? token_layers_raw : 1

    dim_fourier = size(state["structure_module.score_model.single_conditioner.fourier_embed.proj.weight"], 1)
    gaussian_random_3d_encoding_dim = size(state["structure_module.score_model.atom_attention_encoder.r_to_q_trans.weight"], 2) - 3

    use_templates = any(startswith.(state_keys, "template_module."))
    use_token_distances = any(startswith.(state_keys, "token_distance_module."))
    bond_type_feature = any(startswith.(state_keys, "token_bonds_type."))
    msa_use_paired_feature = size(state["msa_module.msa_proj.weight"], 2) == (33 + 2 + 1)

    token_distance_dim = use_token_distances ? size(state["token_distance_module.z_proj.weight"], 1) : token_z
    template_dim = use_templates ? size(state["template_module.z_proj.weight"], 1) : token_z
    token_distance_num_bins = use_token_distances ? Int(size(state["token_distance_module.a_proj.weight"], 2) - 4 * token_z) : 0
    template_num_bins = use_templates ? Int(size(state["template_module.a_proj.weight"], 2) - (2 * 33 + 5)) : 0

    confidence_model_args = _infer_confidence_model_args(state, state_keys, token_z)
    affinity_model_args = _infer_affinity_model_args(state, state_keys, "affinity_module", token_s, token_z)
    affinity_model_args1 = _infer_affinity_model_args(state, state_keys, "affinity_module1", token_s, token_z)
    affinity_model_args2 = _infer_affinity_model_args(state, state_keys, "affinity_module2", token_s, token_z)
    affinity_ensemble = !isnothing(affinity_model_args1) && !isnothing(affinity_model_args2)

    return Dict(
        :atom_s => atom_s,
        :atom_z => atom_z,
        :token_s => token_s,
        :token_z => token_z,
        :num_bins => num_bins,
        :atom_feature_dim => atom_feature_dim,
        :atoms_per_window_queries => 32,
        :atoms_per_window_keys => 128,
        :use_miniformer => false,
        :bond_type_feature => bond_type_feature,
        :use_templates => use_templates,
        :use_token_distances => use_token_distances,
        :pairformer_blocks => pairformer_blocks,
        :pairformer_heads => pairformer_heads,
        :pairwise_head_width => pairwise_head_width,
        :pairwise_num_heads => pairwise_num_heads,
        :msa_blocks => msa_blocks,
        :msa_s => size(state["msa_module.s_proj.weight"], 1),
        :msa_use_paired_feature => msa_use_paired_feature,
        :template_blocks => template_blocks,
        :template_dim => template_dim,
        :template_num_bins => template_num_bins,
        :token_distance_blocks => token_distance_blocks,
        :token_distance_dim => token_distance_dim,
        :token_distance_num_bins => token_distance_num_bins,
        :atom_encoder_depth => atom_encoder_depth,
        :atom_encoder_heads => atom_encoder_heads,
        :score_atom_encoder_depth => atom_enc_depth_diff,
        :score_atom_encoder_heads => atom_enc_heads_diff,
        :score_atom_decoder_depth => atom_dec_depth_diff,
        :score_atom_decoder_heads => atom_dec_heads_diff,
        :token_transformer_depth => token_transformer_depth,
        :token_transformer_heads => 16,
        :token_layers => token_layers,
        :dim_fourier => dim_fourier,
        :gaussian_random_3d_encoding_dim => gaussian_random_3d_encoding_dim,
        :confidence_model_args => isnothing(confidence_model_args) ? Dict{Symbol,Any}() : confidence_model_args,
        :affinity_ensemble => affinity_ensemble,
        :affinity_mw_correction => affinity_ensemble,
        :affinity_model_args => isnothing(affinity_model_args) ? Dict{Symbol,Any}() : affinity_model_args,
        :affinity_model_args1 => isnothing(affinity_model_args1) ? Dict{Symbol,Any}() : affinity_model_args1,
        :affinity_model_args2 => isnothing(affinity_model_args2) ? Dict{Symbol,Any}() : affinity_model_args2,
    )
end

function _build_model_from_config(cfg::AbstractDict{Symbol, <:Any}; confidence_prediction::Bool=false, affinity_prediction::Bool=false)
    return BoltzModel(
        cfg[:atom_s],
        cfg[:atom_z],
        cfg[:token_s],
        cfg[:token_z],
        cfg[:num_bins];
        embedder_args=Dict(
            :atoms_per_window_queries => cfg[:atoms_per_window_queries],
            :atoms_per_window_keys => cfg[:atoms_per_window_keys],
            :atom_feature_dim => cfg[:atom_feature_dim],
            :atom_encoder_depth => cfg[:atom_encoder_depth],
            :atom_encoder_heads => cfg[:atom_encoder_heads],
            :add_method_conditioning => true,
            :add_modified_flag => true,
            :add_cyclic_flag => true,
            :add_mol_type_feat => true,
            :add_ph_flag => false,
            :add_temp_flag => false,
            :add_design_mask_flag => true,
            :add_binding_specification => true,
            :add_ss_specification => true,
            :conditioning_cutoff_min => 4.0,
            :conditioning_cutoff_max => 20.0,
        ),
        msa_args=Dict(
            :msa_s => cfg[:msa_s],
            :msa_blocks => cfg[:msa_blocks],
            :msa_dropout => 0.15,
            :z_dropout => 0.25,
            :miniformer_blocks => false,
            :pairwise_head_width => cfg[:pairwise_head_width],
            :pairwise_num_heads => cfg[:pairwise_num_heads],
            :use_paired_feature => cfg[:msa_use_paired_feature],
            :activation_checkpointing => false,
        ),
        pairformer_args=Dict(
            :num_blocks => cfg[:pairformer_blocks],
            :num_heads => cfg[:pairformer_heads],
            :dropout => 0.25,
            :post_layer_norm => false,
            :pairwise_head_width => cfg[:pairwise_head_width],
            :pairwise_num_heads => cfg[:pairwise_num_heads],
            :activation_checkpointing => false,
        ),
        score_model_args=Dict(
            :sigma_data => 16.0,
            :dim_fourier => cfg[:dim_fourier],
            :atom_encoder_depth => cfg[:score_atom_encoder_depth],
            :atom_encoder_heads => cfg[:score_atom_encoder_heads],
            :token_layers => cfg[:token_layers],
            :token_transformer_depth => cfg[:token_transformer_depth],
            :token_transformer_heads => cfg[:token_transformer_heads],
            :atom_decoder_depth => cfg[:score_atom_decoder_depth],
            :atom_decoder_heads => cfg[:score_atom_decoder_heads],
            :conditioning_transition_layers => 2,
            :gaussian_random_3d_encoding_dim => cfg[:gaussian_random_3d_encoding_dim],
            :transformer_post_ln => false,
            :activation_checkpointing => false,
        ),
        diffusion_process_args=Dict(
            :sigma_min => 0.0004,
            :sigma_max => 160.0,
            :sigma_data => 16.0,
            :rho => 7.0,
            :P_mean => -1.2,
            :P_std => 1.5,
            :gamma_0 => 0.8,
            :gamma_min => 1.0,
            :noise_scale => 0.95,
            :step_scale => 1.8,
            :mse_rotational_alignment => true,
            :coordinate_augmentation => true,
            :alignment_reverse_diff => true,
            :synchronize_sigmas => false,
            :sampling_schedule => "dilated",
            :time_dilation => 2.667,
            :time_dilation_start => 0.6,
            :time_dilation_end => 0.8,
        ),
        token_distance_args=Dict(
            :token_distance_dim => cfg[:token_distance_dim],
            :token_distance_blocks => cfg[:token_distance_blocks],
            :pairwise_head_width => cfg[:pairwise_head_width],
            :pairwise_num_heads => cfg[:pairwise_num_heads],
            :min_dist => 3.25,
            :max_dist => 50.75,
            :num_bins => cfg[:token_distance_num_bins],
            :distance_gaussian_dim => 32,
            :miniformer_blocks => false,
            :use_token_distance_feats => true,
        ),
        template_args=Dict(
            :template_dim => cfg[:template_dim],
            :template_blocks => cfg[:template_blocks],
            :pairwise_head_width => cfg[:pairwise_head_width],
            :pairwise_num_heads => cfg[:pairwise_num_heads],
            :min_dist => 3.25,
            :max_dist => 50.75,
            :num_bins => cfg[:template_num_bins],
            :miniformer_blocks => false,
        ),
        use_miniformer=cfg[:use_miniformer],
        bond_type_feature=cfg[:bond_type_feature],
        use_templates=cfg[:use_templates],
        use_token_distances=cfg[:use_token_distances],
        use_kernels=false,
        confidence_prediction=confidence_prediction,
        confidence_model_args=get(cfg, :confidence_model_args, Dict{Symbol,Any}()),
        affinity_prediction=affinity_prediction,
        affinity_ensemble=get(cfg, :affinity_ensemble, false),
        affinity_mw_correction=get(cfg, :affinity_mw_correction, false),
        affinity_model_args=get(cfg, :affinity_model_args, Dict{Symbol,Any}()),
        affinity_model_args1=get(cfg, :affinity_model_args1, Dict{Symbol,Any}()),
        affinity_model_args2=get(cfg, :affinity_model_args2, Dict{Symbol,Any}()),
    )
end

function load_model_from_state(
    state::AbstractDict{String, <:AbstractArray};
    confidence_prediction::Bool=false,
    affinity_prediction::Bool=false,
)
    cfg = infer_config(state)
    model = _build_model_from_config(cfg; confidence_prediction=confidence_prediction, affinity_prediction=affinity_prediction)
    missing = load_params!(model, state)
    return model, cfg, missing
end

function load_model_from_safetensors(
    path::AbstractString;
    confidence_prediction::Bool=false,
    affinity_prediction::Bool=false,
)
    resolved_path = resolve_weights_path(path)
    state = SafeTensors.load_safetensors(resolved_path)
    return load_model_from_state(
        state;
        confidence_prediction=confidence_prediction,
        affinity_prediction=affinity_prediction,
    )
end
