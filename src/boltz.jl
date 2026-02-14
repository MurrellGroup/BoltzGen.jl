using Onion

const BGLayerNorm = Onion.BGLayerNorm

symbolize_keys(d::Dict) = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d)

# GPU→CPU boundary helpers for confidence/affinity modules
# (these modules use CPU-only operations: loops, sortperm in compute_ptms, Matrix(I,...), etc.)
_to_cpu(x::Array) = x
_to_cpu(x::AbstractArray) = Array(x)
_to_cpu(x) = x
_feats_to_cpu(feats::Dict) = Dict{String,Any}(k => _to_cpu(v) for (k, v) in feats)
_module_to_cpu(m) = Onion.Flux.cpu(m)

@concrete struct BoltzModel <: Onion.Layer
    atom_s::Int
    atom_z::Int
    token_s::Int
    token_z::Int
    num_bins::Int
    use_miniformer::Bool
    bond_type_feature::Bool
    use_templates::Bool
    use_token_distances::Bool
    use_kernels::Bool
    confidence_prediction::Bool
    affinity_prediction::Bool
    affinity_ensemble::Bool
    affinity_mw_correction::Bool
    input_embedder
    s_init
    z_init_1
    z_init_2
    rel_pos
    token_bonds
    token_bonds_type
    contact_conditioning
    s_norm
    z_norm
    s_recycle
    z_recycle
    template_module
    token_distance_module
    msa_module
    pairformer_module
    distogram_module
    diffusion_conditioning
    structure_module
    confidence_module
    affinity_module
    affinity_module1
    affinity_module2
end

@layer BoltzModel

function BoltzModel(
    atom_s::Int,
    atom_z::Int,
    token_s::Int,
    token_z::Int,
    num_bins::Int;
    embedder_args=Dict(),
    msa_args=Dict(),
    pairformer_args=Dict(),
    score_model_args=Dict(),
    diffusion_process_args=Dict(),
    diffusion_loss_args=Dict(),
    token_distance_args=nothing,
    template_args=nothing,
    use_miniformer::Bool=true,
    bond_type_feature::Bool=false,
    use_templates::Bool=false,
    use_token_distances::Bool=false,
    use_kernels::Bool=false,
    confidence_prediction::Bool=false,
    confidence_model_args=Dict(),
    affinity_prediction::Bool=false,
    affinity_model_args=Dict(),
    affinity_ensemble::Bool=false,
    affinity_model_args1=Dict(),
    affinity_model_args2=Dict(),
    affinity_mw_correction::Bool=false,
)
    embedder_args = symbolize_keys(embedder_args)
    msa_args = symbolize_keys(msa_args)
    pairformer_args = symbolize_keys(pairformer_args)
    score_model_args = symbolize_keys(score_model_args)
    diffusion_process_args = symbolize_keys(diffusion_process_args)
    diffusion_loss_args = symbolize_keys(diffusion_loss_args)
    token_distance_args = token_distance_args === nothing ? Dict{Symbol,Any}() : symbolize_keys(token_distance_args)
    template_args = template_args === nothing ? Dict{Symbol,Any}() : symbolize_keys(template_args)
    confidence_model_args = symbolize_keys(confidence_model_args)
    affinity_model_args = symbolize_keys(affinity_model_args)
    affinity_model_args1 = symbolize_keys(affinity_model_args1)
    affinity_model_args2 = symbolize_keys(affinity_model_args2)
    input_embedder = InputEmbedder(
        atom_s,
        atom_z,
        token_s,
        token_z,
        get(embedder_args, :atoms_per_window_queries, 32),
        get(embedder_args, :atoms_per_window_keys, 128),
        get(embedder_args, :atom_feature_dim, 128);
        atom_encoder_depth=get(embedder_args, :atom_encoder_depth, 3),
        atom_encoder_heads=get(embedder_args, :atom_encoder_heads, 4),
        activation_checkpointing=get(embedder_args, :activation_checkpointing, false),
        add_method_conditioning=get(embedder_args, :add_method_conditioning, false),
        add_modified_flag=get(embedder_args, :add_modified_flag, false),
        add_cyclic_flag=get(embedder_args, :add_cyclic_flag, false),
        add_mol_type_feat=get(embedder_args, :add_mol_type_feat, false),
        add_ph_flag=get(embedder_args, :add_ph_flag, false),
        add_temp_flag=get(embedder_args, :add_temp_flag, false),
        add_design_mask_flag=get(embedder_args, :add_design_mask_flag, false),
        add_binding_specification=get(embedder_args, :add_binding_specification, false),
        add_ss_specification=get(embedder_args, :add_ss_specification, false),
    )

    s_init = LinearNoBias(token_s, token_s)
    z_init_1 = LinearNoBias(token_s, token_z)
    z_init_2 = LinearNoBias(token_s, token_z)
    Onion.torch_linear_init!(s_init.weight)
    Onion.torch_linear_init!(z_init_1.weight)
    Onion.torch_linear_init!(z_init_2.weight)

    rel_pos = RelativePositionEncoder(token_z)
    token_bonds = LinearNoBias(1, token_z)
    Onion.torch_linear_init!(token_bonds.weight)

    token_bonds_type = bond_type_feature ? BGEmbedding(length(bond_types) + 1, token_z; init=:torch) : nothing

    contact_conditioning = ContactConditioning(; token_z=token_z, cutoff_min=Float32(get(embedder_args, :conditioning_cutoff_min, 4.0)), cutoff_max=Float32(get(embedder_args, :conditioning_cutoff_max, 20.0)))

    s_norm = BGLayerNorm(token_s; eps=1f-5)
    z_norm = BGLayerNorm(token_z; eps=1f-5)

    s_recycle = LinearNoBias(token_s, token_s)
    z_recycle = LinearNoBias(token_z, token_z)
    Onion.torch_linear_init!(s_recycle.weight)
    Onion.torch_linear_init!(z_recycle.weight)

    template_module = use_templates ? TemplateModule(token_z, get(template_args, :template_dim, token_z), get(template_args, :template_blocks, 1);
        dropout=Float32(get(template_args, :dropout, 0.25)),
        pairwise_head_width=get(template_args, :pairwise_head_width, 32),
        pairwise_num_heads=get(template_args, :pairwise_num_heads, 4),
        post_layer_norm=get(template_args, :post_layer_norm, false),
        activation_checkpointing=get(template_args, :activation_checkpointing, false),
        min_dist=Float32(get(template_args, :min_dist, 3.25)),
        max_dist=Float32(get(template_args, :max_dist, 50.75)),
        num_bins=get(template_args, :num_bins, 38),
        miniformer_blocks=get(template_args, :miniformer_blocks, false),
    ) : nothing

    token_distance_module = use_token_distances ? TokenDistanceModule(token_z, get(token_distance_args, :token_distance_dim, token_z), get(token_distance_args, :token_distance_blocks, 1);
        dropout=Float32(get(token_distance_args, :dropout, 0.25)),
        pairwise_head_width=get(token_distance_args, :pairwise_head_width, 32),
        pairwise_num_heads=get(token_distance_args, :pairwise_num_heads, 4),
        post_layer_norm=get(token_distance_args, :post_layer_norm, false),
        activation_checkpointing=get(token_distance_args, :activation_checkpointing, false),
        min_dist=Float32(get(token_distance_args, :min_dist, 3.25)),
        max_dist=Float32(get(token_distance_args, :max_dist, 50.75)),
        num_bins=get(token_distance_args, :num_bins, 38),
        distance_gaussian_dim=get(token_distance_args, :distance_gaussian_dim, 32),
        miniformer_blocks=get(token_distance_args, :miniformer_blocks, false),
        use_token_distance_feats=get(token_distance_args, :use_token_distance_feats, true),
    ) : nothing

    msa_module = MSAModule(
        get(msa_args, :msa_s, token_s),
        token_z,
        token_s,
        get(msa_args, :msa_blocks, 1),
        Float32(get(msa_args, :msa_dropout, 0.15)),
        Float32(get(msa_args, :z_dropout, 0.25));
        miniformer_blocks=get(msa_args, :miniformer_blocks, true),
        pairwise_head_width=get(msa_args, :pairwise_head_width, 32),
        pairwise_num_heads=get(msa_args, :pairwise_num_heads, 4),
        activation_checkpointing=get(msa_args, :activation_checkpointing, false),
        use_paired_feature=get(msa_args, :use_paired_feature, false),
    )

    pairformer_module = Onion.PairformerModule(
        token_s,
        token_z,
        get(pairformer_args, :num_blocks, 1);
        num_heads=get(pairformer_args, :num_heads, 16),
        dropout=Float32(get(pairformer_args, :dropout, 0.25)),
        pairwise_head_width=get(pairformer_args, :pairwise_head_width, 32),
        pairwise_num_heads=get(pairformer_args, :pairwise_num_heads, 4),
        post_layer_norm=get(pairformer_args, :post_layer_norm, false),
        activation_checkpointing=get(pairformer_args, :activation_checkpointing, false),
    )

    distogram_module = DistogramModule(token_z, num_bins)

    diffusion_conditioning = DiffusionConditioning(
        token_s,
        token_z,
        atom_s,
        atom_z;
        atoms_per_window_queries=get(embedder_args, :atoms_per_window_queries, 32),
        atoms_per_window_keys=get(embedder_args, :atoms_per_window_keys, 128),
        atom_encoder_depth=get(score_model_args, :atom_encoder_depth, 3),
        atom_encoder_heads=get(score_model_args, :atom_encoder_heads, 4),
        token_transformer_depth=get(score_model_args, :token_transformer_depth, 6),
        token_transformer_heads=get(score_model_args, :token_transformer_heads, 8),
        atom_decoder_depth=get(score_model_args, :atom_decoder_depth, 3),
        atom_decoder_heads=get(score_model_args, :atom_decoder_heads, 4),
        atom_feature_dim=get(embedder_args, :atom_feature_dim, 128),
        conditioning_transition_layers=get(score_model_args, :conditioning_transition_layers, 2),
    )

    score_model_args_full = Dict{Symbol,Any}(
        :token_s => token_s,
        :atom_s => atom_s,
        :atoms_per_window_queries => get(embedder_args, :atoms_per_window_queries, 32),
        :atoms_per_window_keys => get(embedder_args, :atoms_per_window_keys, 128),
        :sigma_data => Float32(get(diffusion_process_args, :sigma_data, 16.0)),
        :dim_fourier => get(score_model_args, :dim_fourier, 256),
        :atom_encoder_depth => get(score_model_args, :atom_encoder_depth, 3),
        :atom_encoder_heads => get(score_model_args, :atom_encoder_heads, 4),
        :token_transformer_depth => get(score_model_args, :token_transformer_depth, 6),
        :token_transformer_heads => get(score_model_args, :token_transformer_heads, 8),
        :atom_decoder_depth => get(score_model_args, :atom_decoder_depth, 3),
        :atom_decoder_heads => get(score_model_args, :atom_decoder_heads, 4),
        :conditioning_transition_layers => get(score_model_args, :conditioning_transition_layers, 2),
        :gaussian_random_3d_encoding_dim => get(score_model_args, :gaussian_random_3d_encoding_dim, 0),
        :transformer_post_ln => get(score_model_args, :transformer_post_ln, false),
        :tfmr_s => get(score_model_args, :tfmr_s, nothing),
        :predict_res_type => get(score_model_args, :predict_res_type, false),
        :use_qk_norm => get(score_model_args, :use_qk_norm, false),
    )

    structure_module = AtomDiffusion(score_model_args_full;
        num_sampling_steps=get(diffusion_process_args, :num_sampling_steps, 5),
        sigma_min=Float32(get(diffusion_process_args, :sigma_min, 0.0004)),
        sigma_max=Float32(get(diffusion_process_args, :sigma_max, 160.0)),
        sigma_data=Float32(get(diffusion_process_args, :sigma_data, 16.0)),
        rho=Float32(get(diffusion_process_args, :rho, 7.0)),
        P_mean=Float32(get(diffusion_process_args, :P_mean, -1.2)),
        P_std=Float32(get(diffusion_process_args, :P_std, 1.5)),
        gamma_0=Float32(get(diffusion_process_args, :gamma_0, 0.8)),
        gamma_min=Float32(get(diffusion_process_args, :gamma_min, 1.0)),
        noise_scale=Float32(get(diffusion_process_args, :noise_scale, 1.003)),
        step_scale=Float32(get(diffusion_process_args, :step_scale, 1.5)),
        coordinate_augmentation=get(diffusion_process_args, :coordinate_augmentation, true),
        coordinate_augmentation_inference=get(diffusion_process_args, :coordinate_augmentation_inference, nothing),
        mse_rotational_alignment=get(diffusion_process_args, :mse_rotational_alignment, false),
        alignment_reverse_diff=get(diffusion_process_args, :alignment_reverse_diff, false),
        synchronize_sigmas=get(diffusion_process_args, :synchronize_sigmas, false),
        second_order_correction=get(diffusion_process_args, :second_order_correction, false),
        pass_resolved_mask_diff_train=get(diffusion_process_args, :pass_resolved_mask_diff_train, false),
        sampling_schedule=get(diffusion_process_args, :sampling_schedule, "af3"),
        time_dilation=Float32(get(diffusion_process_args, :time_dilation, 1.0)),
        time_dilation_start=Float32(get(diffusion_process_args, :time_dilation_start, 0.6)),
        time_dilation_end=Float32(get(diffusion_process_args, :time_dilation_end, 0.8)),
        noise_scale_function=get(diffusion_process_args, :noise_scale_function, "constant"),
        step_scale_function=get(diffusion_process_args, :step_scale_function, "constant"),
        min_noise_scale=Float32(get(diffusion_process_args, :min_noise_scale, 1.0)),
        max_noise_scale=Float32(get(diffusion_process_args, :max_noise_scale, 1.0)),
        noise_scale_alpha=Float32(get(diffusion_process_args, :noise_scale_alpha, 1.0)),
        noise_scale_beta=Float32(get(diffusion_process_args, :noise_scale_beta, 1.0)),
        min_step_scale=Float32(get(diffusion_process_args, :min_step_scale, 1.0)),
        max_step_scale=Float32(get(diffusion_process_args, :max_step_scale, 1.0)),
        step_scale_alpha=Float32(get(diffusion_process_args, :step_scale_alpha, 1.0)),
        step_scale_beta=Float32(get(diffusion_process_args, :step_scale_beta, 1.0)),
        step_scale_random=get(diffusion_process_args, :step_scale_random, nothing),
        pred_threshold=get(diffusion_process_args, :pred_threshold, nothing),
    )

    confidence_module = nothing
    if confidence_prediction
        conf_pairformer_args = symbolize_keys(get(confidence_model_args, :pairformer_args, Dict()))
        conf_head_args = symbolize_keys(get(confidence_model_args, :confidence_args, Dict()))
        confidence_module = ConfidenceModule(
            token_s,
            token_z;
            pairformer_args=conf_pairformer_args,
            num_dist_bins=get(confidence_model_args, :num_dist_bins, 64),
            token_level_confidence=get(confidence_model_args, :token_level_confidence, true),
            max_dist=get(confidence_model_args, :max_dist, 22),
            add_s_to_z_prod=get(confidence_model_args, :add_s_to_z_prod, false),
            confidence_args=conf_head_args,
            return_latent_feats=get(confidence_model_args, :return_latent_feats, false),
            conditioning_cutoff_min=get(confidence_model_args, :conditioning_cutoff_min, get(embedder_args, :conditioning_cutoff_min, 4.0)),
            conditioning_cutoff_max=get(confidence_model_args, :conditioning_cutoff_max, get(embedder_args, :conditioning_cutoff_max, 20.0)),
            bond_type_feature=bond_type_feature,
            add_z_input_to_z=get(confidence_model_args, :add_z_input_to_z, nothing),
            add_s_input_to_s=get(confidence_model_args, :add_s_input_to_s, false),
            no_update_s=get(confidence_model_args, :no_update_s, false),
        )
    end

    affinity_module = nothing
    affinity_module1 = nothing
    affinity_module2 = nothing
    if affinity_prediction
        if affinity_ensemble
            aff_pairformer_args1 = symbolize_keys(get(affinity_model_args1, :pairformer_args, Dict()))
            aff_transformer_args1 = symbolize_keys(get(affinity_model_args1, :transformer_args, Dict()))
            affinity_module1 = AffinityModule(
                token_s,
                token_z;
                pairformer_args=aff_pairformer_args1,
                transformer_args=aff_transformer_args1,
                num_dist_bins=get(affinity_model_args1, :num_dist_bins, 64),
                max_dist=get(affinity_model_args1, :max_dist, 22),
                use_cross_transformer=get(affinity_model_args1, :use_cross_transformer, false),
                groups=get(affinity_model_args1, :groups, Dict()),
            )
            aff_pairformer_args2 = symbolize_keys(get(affinity_model_args2, :pairformer_args, Dict()))
            aff_transformer_args2 = symbolize_keys(get(affinity_model_args2, :transformer_args, Dict()))
            affinity_module2 = AffinityModule(
                token_s,
                token_z;
                pairformer_args=aff_pairformer_args2,
                transformer_args=aff_transformer_args2,
                num_dist_bins=get(affinity_model_args2, :num_dist_bins, 64),
                max_dist=get(affinity_model_args2, :max_dist, 22),
                use_cross_transformer=get(affinity_model_args2, :use_cross_transformer, false),
                groups=get(affinity_model_args2, :groups, Dict()),
            )
        else
            aff_pairformer_args = symbolize_keys(get(affinity_model_args, :pairformer_args, Dict()))
            aff_transformer_args = symbolize_keys(get(affinity_model_args, :transformer_args, Dict()))
            affinity_module = AffinityModule(
                token_s,
                token_z;
                pairformer_args=aff_pairformer_args,
                transformer_args=aff_transformer_args,
                num_dist_bins=get(affinity_model_args, :num_dist_bins, 64),
                max_dist=get(affinity_model_args, :max_dist, 22),
                use_cross_transformer=get(affinity_model_args, :use_cross_transformer, false),
                groups=get(affinity_model_args, :groups, Dict()),
            )
        end
    end

    return BoltzModel(
        atom_s,
        atom_z,
        token_s,
        token_z,
        num_bins,
        use_miniformer,
        bond_type_feature,
        use_templates,
        use_token_distances,
        use_kernels,
        confidence_prediction,
        affinity_prediction,
        affinity_ensemble,
        affinity_mw_correction,
        input_embedder,
        s_init,
        z_init_1,
        z_init_2,
        rel_pos,
        token_bonds,
        token_bonds_type,
        contact_conditioning,
        s_norm,
        z_norm,
        s_recycle,
        z_recycle,
        template_module,
        token_distance_module,
        msa_module,
        pairformer_module,
        distogram_module,
        diffusion_conditioning,
        structure_module,
        confidence_module,
        affinity_module,
        affinity_module1,
        affinity_module2,
    )
end

function boltz_forward(model::BoltzModel, feats; recycling_steps::Int=0, num_sampling_steps=nothing, diffusion_samples::Int=1, step_scale=nothing, noise_scale=nothing, sampling_schedule=nothing, time_dilation=nothing, time_dilation_start=nothing, time_dilation_end=nothing, atom_coords_init=nothing, return_z_feats::Bool=false, run_confidence_sequentially::Bool=false, inference_logging::Bool=false)
    s_inputs = model.input_embedder(feats)

    s_init = model.s_init(s_inputs)
    z_init = reshape(model.z_init_1(s_inputs), model.token_z, size(s_inputs,2), 1, size(s_inputs,3)) .+
        reshape(model.z_init_2(s_inputs), model.token_z, 1, size(s_inputs,2), size(s_inputs,3))

    relative_position_encoding = model.rel_pos(feats)
    z_init = z_init .+ relative_position_encoding
    z_init = z_init .+ model.token_bonds(feats["token_bonds"])
    if model.bond_type_feature && model.token_bonds_type !== nothing
        z_init = z_init .+ model.token_bonds_type(feats["type_bonds"])
    end
    z_init = z_init .+ model.contact_conditioning(feats)

    s = Onion.zeros_like(s_init)
    z = Onion.zeros_like(z_init)

    mask = feats["token_pad_mask"]
    pair_mask = reshape(mask, size(mask,1), 1, size(mask,2)) .* reshape(mask, 1, size(mask,1), size(mask,2))

    for i in 1:(recycling_steps + 1)
        if inference_logging
            println("Trunk recycle ", i, "/", recycling_steps + 1)
        end
        s = s_init .+ model.s_recycle(model.s_norm(s))
        z = z_init .+ model.z_recycle(model.z_norm(z))

        if model.use_token_distances && model.token_distance_module !== nothing
            z = z .+ model.token_distance_module(z, feats, pair_mask, relative_position_encoding; use_kernels=model.use_kernels)
        end
        if model.use_templates && model.template_module !== nothing
            z = z .+ model.template_module(z, feats, pair_mask; use_kernels=model.use_kernels)
        end

        z = z .+ model.msa_module(z, s_inputs, feats; use_kernels=model.use_kernels)

        s, z = model.pairformer_module(s, z, mask, pair_mask; use_kernels=model.use_kernels)
    end

    pdistogram = model.distogram_module(z)

    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = model.diffusion_conditioning(
        s_trunk=s,
        z_trunk=z,
        relative_position_encoding=relative_position_encoding,
        feats=feats,
    )
    diffusion_conditioning = Dict(
        "q" => q,
        "c" => c,
        "to_keys" => to_keys,
        "atom_enc_bias" => atom_enc_bias,
        "atom_dec_bias" => atom_dec_bias,
        "token_trans_bias" => token_trans_bias,
    )

    if inference_logging
        println("Starting diffusion sampling")
    end
    struct_out = sample(model.structure_module;
        s_trunk=s,
        s_inputs=s_inputs,
        feats=feats,
        atom_mask=feats["atom_pad_mask"],
        num_sampling_steps=num_sampling_steps,
        multiplicity=diffusion_samples,
        diffusion_conditioning=diffusion_conditioning,
        step_scale=step_scale,
        noise_scale=noise_scale,
        sampling_schedule=sampling_schedule,
        time_dilation=time_dilation,
        time_dilation_start=time_dilation_start,
        time_dilation_end=time_dilation_end,
        atom_coords_init=atom_coords_init,
        inference_logging=inference_logging,
    )

    out = Dict{String,Any}(
        "pdistogram" => pdistogram,
    )
    for (k, v) in struct_out
        out[k] = v
    end

    if model.confidence_prediction && model.confidence_module !== nothing
        # Confidence utils use CPU-only operations (loops, sortperm in compute_ptms, etc.)
        # Move data and module weights to CPU for this computation.
        conf_cpu = _module_to_cpu(model.confidence_module)
        feats_cpu = _feats_to_cpu(feats)
        pred_dist_logits = _to_cpu(dropdims(pdistogram; dims=4))
        conf_out = conf_cpu(
            _to_cpu(s_inputs),
            _to_cpu(s),
            _to_cpu(z),
            _to_cpu(struct_out["sample_atom_coords"]),
            feats_cpu,
            pred_dist_logits;
            multiplicity=diffusion_samples,
            run_sequentially=run_confidence_sequentially,
            use_kernels=model.use_kernels,
        )
        for (k, v) in conf_out
            out[k] = v
        end
    end

    if model.affinity_prediction && (model.affinity_module !== nothing || model.affinity_module1 !== nothing)
        # Compute masks on the current device (GPU-compatible ops)
        pad_token_mask = feats["token_pad_mask"]
        rec_mask = (feats["mol_type"] .== chain_type_ids["PROTEIN"]) .* pad_token_mask
        lig_mask = (feats["affinity_token_mask"] .> 0) .* pad_token_mask
        cross_pair_mask = reshape(lig_mask, size(lig_mask, 1), 1, size(lig_mask, 2)) .* reshape(rec_mask, 1, size(rec_mask, 1), size(rec_mask, 2)) .+
            reshape(rec_mask, size(rec_mask, 1), 1, size(rec_mask, 2)) .* reshape(lig_mask, 1, size(lig_mask, 1), size(lig_mask, 2)) .+
            reshape(lig_mask, size(lig_mask, 1), 1, size(lig_mask, 2)) .* reshape(lig_mask, 1, size(lig_mask, 1), size(lig_mask, 2))
        z_affinity = z .* reshape(cross_pair_mask, 1, size(cross_pair_mask, 1), size(cross_pair_mask, 2), size(cross_pair_mask, 3))

        best_idx = 1
        if haskey(out, "iptm")
            iptm = out["iptm"]
            best_idx = argmax(iptm)
        end

        coords_affinity = struct_out["sample_atom_coords"][:, :, best_idx:best_idx]
        s_inputs_aff = model.input_embedder(feats; affinity=true)

        # Affinity module uses CPU-only operations (Matrix{Float32}(I,...) in AffinityHeadsTransformer)
        # Move data and module weights to CPU.
        feats_cpu_aff = @isdefined(feats_cpu) ? feats_cpu : _feats_to_cpu(feats)
        z_aff_cpu = _to_cpu(z_affinity)
        coords_aff_cpu = _to_cpu(coords_affinity)
        s_inputs_aff_cpu = _to_cpu(s_inputs_aff)

        if model.affinity_ensemble
            aff1_cpu = _module_to_cpu(model.affinity_module1)
            dict_out_affinity1 = aff1_cpu(
                s_inputs_aff_cpu,
                z_aff_cpu,
                coords_aff_cpu,
                feats_cpu_aff;
                multiplicity=1,
                use_kernels=model.use_kernels,
            )
            prob1 = NNlib.sigmoid.(dict_out_affinity1["affinity_logits_binary"])
            aff2_cpu = _module_to_cpu(model.affinity_module2)
            dict_out_affinity2 = aff2_cpu(
                s_inputs_aff_cpu,
                z_aff_cpu,
                coords_aff_cpu,
                feats_cpu_aff;
                multiplicity=1,
                use_kernels=model.use_kernels,
            )
            prob2 = NNlib.sigmoid.(dict_out_affinity2["affinity_logits_binary"])

            affinity_pred_value = (dict_out_affinity1["affinity_pred_value"] .+ dict_out_affinity2["affinity_pred_value"]) ./ 2
            affinity_probability_binary = (prob1 .+ prob2) ./ 2

            if model.affinity_mw_correction
                model_coef = 1.03525938f0
                mw_coef = -0.59992683f0
                bias = 2.83288489f0
                mw = _to_cpu(feats["affinity_mw"]) .^ 0.3f0
                affinity_pred_value = model_coef .* affinity_pred_value .+ mw_coef .* mw .+ bias
            end

            out["affinity_pred_value"] = affinity_pred_value
            out["affinity_probability_binary"] = affinity_probability_binary
            out["affinity_pred_value1"] = dict_out_affinity1["affinity_pred_value"]
            out["affinity_probability_binary1"] = prob1
            out["affinity_pred_value2"] = dict_out_affinity2["affinity_pred_value"]
            out["affinity_probability_binary2"] = prob2
        else
            aff_cpu = _module_to_cpu(model.affinity_module)
            dict_out_affinity = aff_cpu(
                s_inputs_aff_cpu,
                z_aff_cpu,
                coords_aff_cpu,
                feats_cpu_aff;
                multiplicity=1,
                use_kernels=model.use_kernels,
            )
            out["affinity_pred_value"] = dict_out_affinity["affinity_pred_value"]
            out["affinity_probability_binary"] = NNlib.sigmoid.(dict_out_affinity["affinity_logits_binary"])
        end
    end

    if return_z_feats
        out["s_trunk"] = s
        out["z_trunk"] = z
    end

    return out
end
