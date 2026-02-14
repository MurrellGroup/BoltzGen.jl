using Onion
using NNlib
using Statistics

const BGLayerNorm = Onion.BGLayerNorm

@concrete struct DiffusionModule <: Onion.Layer
    atoms_per_window_queries::Int
    atoms_per_window_keys::Int
    sigma_data
    activation_checkpointing::Bool
    tfmr_s::Int
    single_conditioner
    atom_attention_encoder
    s_to_a_linear
    token_transformer
    a_norm
    atom_attention_decoder
end

@layer DiffusionModule

function DiffusionModule(
    token_s::Int,
    atom_s::Int;
    atoms_per_window_queries::Int=32,
    atoms_per_window_keys::Int=128,
    sigma_data::Float32=16f0,
    dim_fourier::Int=256,
    atom_encoder_depth::Int=3,
    atom_encoder_heads::Int=4,
    token_layers::Int=1,
    token_transformer_depth::Int=6,
    token_transformer_heads::Int=8,
    use_miniformer::Bool=false,
    diffusion_pairformer_args=nothing,
    atom_decoder_depth::Int=3,
    atom_decoder_heads::Int=4,
    conditioning_transition_layers::Int=2,
    activation_checkpointing::Bool=false,
    gaussian_random_3d_encoding_dim::Int=0,
    transformer_post_ln::Bool=false,
    tfmr_s::Union{Nothing,Int}=nothing,
    predict_res_type::Bool=false,
    use_qk_norm::Bool=false,
)
    tfmr_s = tfmr_s === nothing ? 2 * token_s : tfmr_s

    single_conditioner = SingleConditioning(
        sigma_data;
        tfmr_s=tfmr_s,
        token_s=token_s,
        dim_fourier=dim_fourier,
        num_transitions=conditioning_transition_layers,
    )

    atom_attention_encoder = AtomAttentionEncoder(
        atom_s,
        token_s,
        atoms_per_window_queries,
        atoms_per_window_keys;
        atom_encoder_depth=atom_encoder_depth,
        atom_encoder_heads=atom_encoder_heads,
        structure_prediction=true,
        activation_checkpointing=activation_checkpointing,
        gaussian_random_3d_encoding_dim=gaussian_random_3d_encoding_dim,
        transformer_post_layer_norm=transformer_post_ln,
        tfmr_s=tfmr_s,
        use_qk_norm=use_qk_norm,
    )

    s_to_a_linear = (BGLayerNorm(tfmr_s; eps=1f-5), LinearNoBias(tfmr_s, tfmr_s))
    Onion.final_init!(s_to_a_linear[2].weight)

    token_transformer = DiffusionTransformer(; dim=tfmr_s, dim_single_cond=tfmr_s, depth=token_transformer_depth, heads=token_transformer_heads, activation_checkpointing=activation_checkpointing, use_qk_norm=use_qk_norm)

    a_norm = BGLayerNorm(tfmr_s; eps=1f-5)

    atom_attention_decoder = AtomAttentionDecoder(
        atom_s,
        tfmr_s,
        atoms_per_window_queries,
        atoms_per_window_keys;
        atom_decoder_depth=atom_decoder_depth,
        atom_decoder_heads=atom_decoder_heads,
        activation_checkpointing=activation_checkpointing,
        transformer_post_layer_norm=transformer_post_ln,
        predict_res_type=predict_res_type,
        use_qk_norm=use_qk_norm,
    )

    return DiffusionModule(atoms_per_window_queries, atoms_per_window_keys, sigma_data, activation_checkpointing, tfmr_s,
        single_conditioner, atom_attention_encoder, s_to_a_linear, token_transformer, a_norm, atom_attention_decoder)
end

function (dm::DiffusionModule)(; s_inputs, s_trunk, r_noisy, times, feats, diffusion_conditioning, multiplicity::Int=1)
    s, normed_fourier = dm.single_conditioner(times, repeat_interleave_batch(s_trunk, multiplicity), repeat_interleave_batch(s_inputs, multiplicity))

    a, q_skip, c_skip, to_keys = dm.atom_attention_encoder(
        feats=feats,
        q=diffusion_conditioning["q"],
        c=diffusion_conditioning["c"],
        atom_enc_bias=diffusion_conditioning["atom_enc_bias"],
        to_keys=diffusion_conditioning["to_keys"],
        r=r_noisy,
        multiplicity=multiplicity,
    )

    ln, lin = dm.s_to_a_linear
    a = a .+ lin(ln(s))

    mask = repeat_interleave_batch(feats["token_pad_mask"], multiplicity)

    a = dm.token_transformer(a, s; bias=diffusion_conditioning["token_trans_bias"], mask=mask, multiplicity=multiplicity, to_keys=nothing)
    a = dm.a_norm(a)

    r_update, res_type = dm.atom_attention_decoder(
        a=a,
        q=q_skip,
        c=c_skip,
        atom_dec_bias=diffusion_conditioning["atom_dec_bias"],
        feats=feats,
        to_keys=diffusion_conditioning["to_keys"],
        multiplicity=multiplicity,
    )

    return Dict(
        "r_update" => r_update,
        "token_a" => a,
        "res_type" => res_type,
    )
end

@concrete struct AtomDiffusion <: Onion.Layer
    score_model
    sigma_min
    sigma_max
    sigma_data
    rho
    P_mean
    P_std
    pred_sigma_thresh
    num_sampling_steps
    sampling_schedule
    time_dilation
    time_dilation_start
    time_dilation_end
    gamma_0
    gamma_min
    noise_scale
    noise_scale_function
    min_noise_scale
    max_noise_scale
    noise_scale_alpha
    noise_scale_beta
    step_scale
    step_scale_function
    min_step_scale
    max_step_scale
    step_scale_alpha
    step_scale_beta
    step_scale_random
    coordinate_augmentation
    coordinate_augmentation_inference
    mse_rotational_alignment
    alignment_reverse_diff
    synchronize_sigmas
    second_order_correction
    pass_resolved_mask_diff_train
    token_s
end

@layer AtomDiffusion

function AtomDiffusion(
    score_model_args;
    num_sampling_steps::Int=5,
    sigma_min::Float32=0.0004f0,
    sigma_max::Float32=160f0,
    sigma_data::Float32=16f0,
    rho::Float32=7f0,
    P_mean::Float32=-1.2f0,
    P_std::Float32=1.5f0,
    gamma_0::Float32=0.8f0,
    gamma_min::Float32=1.0f0,
    noise_scale=1.003f0,
    step_scale=1.5f0,
    step_scale_random=nothing,
    coordinate_augmentation::Bool=true,
    coordinate_augmentation_inference=nothing,
    mse_rotational_alignment::Bool=false,
    alignment_reverse_diff::Bool=false,
    synchronize_sigmas::Bool=false,
    second_order_correction::Bool=false,
    pass_resolved_mask_diff_train::Bool=false,
    sampling_schedule::String="af3",
    noise_scale_function::String="constant",
    step_scale_function::String="constant",
    min_noise_scale::Float32=1.0f0,
    max_noise_scale::Float32=1.0f0,
    noise_scale_alpha::Float32=1.0f0,
    noise_scale_beta::Float32=1.0f0,
    min_step_scale::Float32=1.0f0,
    max_step_scale::Float32=1.0f0,
    step_scale_alpha::Float32=1.0f0,
    step_scale_beta::Float32=1.0f0,
    time_dilation::Float32=1.0f0,
    time_dilation_start::Float32=0.6f0,
    time_dilation_end::Float32=0.8f0,
    pred_threshold=nothing,
)
    token_s = score_model_args[:token_s]
    atom_s = score_model_args[:atom_s]
    kwargs = copy(score_model_args)
    delete!(kwargs, :token_s)
    delete!(kwargs, :atom_s)
    score_model = DiffusionModule(token_s, atom_s; kwargs...)
    pred_sigma_thresh = pred_threshold === nothing ? Inf32 : sigma_data * exp(P_mean + P_std * Float32(norm_ppf(pred_threshold)))

    return AtomDiffusion(
        score_model,
        sigma_min,
        sigma_max,
        sigma_data,
        rho,
        P_mean,
        P_std,
        pred_sigma_thresh,
        num_sampling_steps,
        sampling_schedule,
        time_dilation,
        time_dilation_start,
        time_dilation_end,
        gamma_0,
        gamma_min,
        noise_scale,
        noise_scale_function,
        min_noise_scale,
        max_noise_scale,
        noise_scale_alpha,
        noise_scale_beta,
        step_scale,
        step_scale_function,
        min_step_scale,
        max_step_scale,
        step_scale_alpha,
        step_scale_beta,
        step_scale_random,
        coordinate_augmentation,
        coordinate_augmentation_inference === nothing ? coordinate_augmentation : coordinate_augmentation_inference,
        mse_rotational_alignment,
        alignment_reverse_diff,
        synchronize_sigmas,
        second_order_correction,
        pass_resolved_mask_diff_train,
        token_s,
    )
end

function c_skip(ad::AtomDiffusion, sigma)
    return (ad.sigma_data^2) ./ (sigma.^2 .+ ad.sigma_data^2)
end

function c_out(ad::AtomDiffusion, sigma)
    return sigma .* ad.sigma_data ./ sqrt.(ad.sigma_data^2 .+ sigma.^2)
end

function c_in(ad::AtomDiffusion, sigma)
    return 1f0 ./ sqrt.(ad.sigma_data^2 .+ sigma.^2)
end

function c_noise(ad::AtomDiffusion, sigma)
    return bg_log(sigma ./ ad.sigma_data) .* 0.25f0
end

function preconditioned_network_forward(ad::AtomDiffusion, noised_atom_coords, sigma, network_condition_kwargs; training::Bool=true)
    batch = size(noised_atom_coords, 3)
    sigma_vec = sigma
    if isa(sigma, Float32) || isa(sigma, Float64)
        sigma_vec = fill!(similar(noised_atom_coords, Float32, batch), Float32(sigma))
    end

    padded_sigma = reshape(sigma_vec, 1, 1, length(sigma_vec))

    if training && ad.pass_resolved_mask_diff_train
        res_mask = network_condition_kwargs["feats"]["atom_resolved_mask"]
        res_mask = repeat_interleave_batch(res_mask, network_condition_kwargs["multiplicity"])
        res_mask = reshape(Float32.(res_mask), 1, size(res_mask,1), size(res_mask,2))
        noised_atom_coords = noised_atom_coords .* res_mask
    end

    net_out = ad.score_model(
        s_inputs=network_condition_kwargs["s_inputs"],
        s_trunk=network_condition_kwargs["s_trunk"],
        r_noisy=c_in(ad, padded_sigma) .* noised_atom_coords,
        times=c_noise(ad, sigma_vec),
        feats=network_condition_kwargs["feats"],
        multiplicity=network_condition_kwargs["multiplicity"],
        diffusion_conditioning=network_condition_kwargs["diffusion_conditioning"],
    )

    denoised_coords = c_skip(ad, padded_sigma) .* noised_atom_coords .+ c_out(ad, padded_sigma) .* net_out["r_update"]

    return denoised_coords, net_out
end

function sample_schedule_af3(ad::AtomDiffusion, num_sampling_steps=nothing)
    num_sampling_steps = default(num_sampling_steps, ad.num_sampling_steps)
    inv_rho = 1f0 / ad.rho
    steps = collect(0:num_sampling_steps-1)
    sigmas = (ad.sigma_max^inv_rho .+ (steps ./ (num_sampling_steps - 1)) .* (ad.sigma_min^inv_rho - ad.sigma_max^inv_rho)).^ad.rho
    sigmas = sigmas .* ad.sigma_data
    sigmas = vcat(sigmas, 0f0)
    return Float32.(sigmas)
end

function sample_schedule_dilated(ad::AtomDiffusion, num_sampling_steps=nothing;
        time_dilation=ad.time_dilation, time_dilation_start=ad.time_dilation_start, time_dilation_end=ad.time_dilation_end)
    num_sampling_steps = default(num_sampling_steps, ad.num_sampling_steps)
    inv_rho = 1f0 / ad.rho
    steps = collect(0:num_sampling_steps-1)
    ts = steps ./ (num_sampling_steps - 1)

    function dilate(ts, start, end_, dilation)
        x = end_ - start
        l = start
        u = 1 - end_
        @assert (dilation - 1) * x <= l + u "dilation too large"
        inv_dilation = 1 / dilation
        ratio = (l + u + (1 - dilation) * x) / (l + u)
        inv_ratio = 1 / ratio
        lprime = l * ratio
        uprime = u * ratio
        xprime = x * dilation

        lower_third = ts .* inv_ratio
        middle_third = (ts .- lprime) .* inv_dilation .+ l
        upper_third = (ts .- (lprime + xprime)) .* inv_ratio .+ l .+ x
        return (ts .< lprime) .* lower_third .+ ((ts .>= lprime) .& (ts .< lprime + xprime)) .* middle_third .+ (ts .>= lprime + xprime) .* upper_third
    end

    dilated_ts = dilate(ts, time_dilation_start, time_dilation_end, time_dilation)
    sigmas = (ad.sigma_max^inv_rho .+ dilated_ts .* (ad.sigma_min^inv_rho - ad.sigma_max^inv_rho)).^ad.rho
    sigmas = sigmas .* ad.sigma_data
    sigmas = vcat(sigmas, 0f0)
    return Float32.(sigmas)
end

function beta_noise_scale_schedule(ad::AtomDiffusion, num_sampling_steps)
    t = range(0f0, 1f0; length=num_sampling_steps)
    beta_weights = [beta_cdf(1f0 - Float64(tt), ad.noise_scale_alpha, ad.noise_scale_beta) for tt in t]
    return Float32.(ad.max_noise_scale .+ (ad.min_noise_scale - ad.max_noise_scale) .* beta_weights)
end

function beta_step_scale_schedule(ad::AtomDiffusion, num_sampling_steps)
    t = range(0f0, 1f0; length=num_sampling_steps)
    beta_weights = [beta_cdf(Float64(tt), ad.step_scale_alpha, ad.step_scale_beta) for tt in t]
    return Float32.(ad.min_step_scale .+ (ad.max_step_scale - ad.min_step_scale) .* beta_weights)
end

function sample(ad::AtomDiffusion; atom_mask, num_sampling_steps=nothing, multiplicity::Int=1, step_scale=nothing, noise_scale=nothing, sampling_schedule=nothing, time_dilation=nothing, time_dilation_start=nothing, time_dilation_end=nothing, atom_coords_init=nothing, inference_logging::Bool=false, network_condition_kwargs...)
    num_sampling_steps = default(num_sampling_steps, ad.num_sampling_steps)

    if Onion.bg_istraining() && ad.step_scale_random !== nothing
        step_scales = fill(Float32(rand(ad.step_scale_random)), num_sampling_steps)
    elseif ad.step_scale_function == "beta"
        step_scales = beta_step_scale_schedule(ad, num_sampling_steps)
    else
        ss = default(step_scale, ad.step_scale)
        ss === nothing && error("step_scale is nothing")
        step_scales = fill(Float32(ss), num_sampling_steps)
    end

    if ad.noise_scale_function == "constant"
        ns = default(noise_scale, ad.noise_scale)
        ns === nothing && error("noise_scale is nothing")
        noise_scales = fill(Float32(ns), num_sampling_steps)
    elseif ad.noise_scale_function == "beta"
        noise_scales = beta_noise_scale_schedule(ad, num_sampling_steps)
    else
        error("Invalid noise scale schedule: $(ad.noise_scale_function)")
    end

    atom_mask_rep = repeat_interleave_batch(atom_mask, multiplicity)
    shape = (3, size(atom_mask_rep,1), size(atom_mask_rep,2))

    eff_schedule = default(sampling_schedule, ad.sampling_schedule)
    if eff_schedule == "af3"
        sigmas = sample_schedule_af3(ad, num_sampling_steps)
    elseif eff_schedule == "dilated"
        # Use caller-provided dilation params if given, else struct defaults
        eff_td = Float32(default(time_dilation, ad.time_dilation))
        eff_tds = Float32(default(time_dilation_start, ad.time_dilation_start))
        eff_tde = Float32(default(time_dilation_end, ad.time_dilation_end))
        sigmas = sample_schedule_dilated(ad, num_sampling_steps;
            time_dilation=eff_td, time_dilation_start=eff_tds, time_dilation_end=eff_tde)
    else
        error("Invalid sampling schedule: $(eff_schedule)")
    end
    gammas = ifelse.(sigmas .> ad.gamma_min, ad.gamma_0, 0f0)

    init_sigma = sigmas[1]
    if atom_coords_init === nothing
        atom_coords = init_sigma .* randn!(similar(atom_mask_rep, Float32, shape))
    else
        atom_coords = Float32.(atom_coords_init)
    end

    coords_traj = [atom_coords]
    x0_coords_traj = []

    for step_idx in 1:num_sampling_steps
        if inference_logging && (step_idx == 1 || step_idx % 5 == 0 || step_idx == num_sampling_steps)
            println("Diffusion step ", step_idx, "/", num_sampling_steps)
        end
        sigma_tm = sigmas[step_idx]
        sigma_t = sigmas[step_idx + 1]
        gamma = gammas[step_idx + 1]
        t_hat = sigma_tm * (1 + gamma)
        noise_var = noise_scales[step_idx]^2 * (t_hat^2 - sigma_tm^2)

        atom_coords = center(atom_coords, atom_mask_rep)
        if ad.coordinate_augmentation_inference
            R, random_tr = compute_random_augmentation(multiplicity, atom_coords)
            atom_coords = rotate_coords(atom_coords, R) .+ random_tr
        end

        eps = noise_scales[step_idx] * sqrt(noise_var) .* randn!(similar(atom_coords, Float32, shape))
        atom_coords_noisy = atom_coords .+ eps

        atom_coords_denoised, net_out = preconditioned_network_forward(
            ad,
            atom_coords_noisy,
            t_hat,
            Dict(
                "s_inputs" => network_condition_kwargs[:s_inputs],
                "s_trunk" => network_condition_kwargs[:s_trunk],
                "feats" => network_condition_kwargs[:feats],
                "multiplicity" => multiplicity,
                "diffusion_conditioning" => network_condition_kwargs[:diffusion_conditioning],
            );
            training=false,
        )

        if ad.alignment_reverse_diff
            atom_coords_noisy = weighted_rigid_align(
                atom_coords_noisy,
                atom_coords_denoised,
                Float32.(atom_mask_rep),
                atom_mask_rep .> 0.5,
            )
        end

        denoised_over_sigma = (atom_coords_noisy .- atom_coords_denoised) ./ t_hat
        atom_coords_next = atom_coords_noisy .+ step_scales[step_idx] .* (sigma_t - t_hat) .* denoised_over_sigma

        push!(coords_traj, atom_coords_next)
        push!(x0_coords_traj, atom_coords_denoised)
        atom_coords = atom_coords_next
    end

    return Dict(
        "sample_atom_coords" => atom_coords,
        "coords_traj" => coords_traj,
        "x0_coords_traj" => x0_coords_traj,
    )
end

function (ad::AtomDiffusion)(; s_inputs, s_trunk, feats, diffusion_conditioning, multiplicity::Int=1)
    atom_coords = feats["coords"]
    atom_mask = feats["atom_pad_mask"]

    atom_coords = center_random_augmentation(atom_coords, atom_mask; augmentation=ad.coordinate_augmentation, centering=true)
    if ad.synchronize_sigmas
        sigmas = ad.sigma_data .* exp.(ad.P_mean .+ ad.P_std .* randn!(similar(atom_coords, Float32, size(atom_coords,3))))
    else
        sigmas = ad.sigma_data .* exp.(ad.P_mean .+ ad.P_std .* randn!(similar(atom_coords, Float32, size(atom_coords,3))))
    end

    padded_sigmas = reshape(sigmas, 1, 1, length(sigmas))
    noise = randn!(similar(atom_coords))
    noised_atom_coords = atom_coords .+ padded_sigmas .* noise

    denoised_atom_coords, net_out = preconditioned_network_forward(
        ad,
        noised_atom_coords,
        sigmas,
        Dict(
            "s_inputs" => s_inputs,
            "s_trunk" => s_trunk,
            "feats" => feats,
            "multiplicity" => multiplicity,
            "diffusion_conditioning" => diffusion_conditioning,
        );
        training=true,
    )

    out = Dict(
        "noised_atom_coords" => noised_atom_coords,
        "denoised_atom_coords" => denoised_atom_coords,
        "sigmas" => sigmas,
        "aligned_true_atom_coords" => atom_coords,
    )
    for (k, v) in net_out
        out[k] = v
    end
    return out
end
