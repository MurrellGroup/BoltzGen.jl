# ── Differentiable Boltz2 forward path ──────────────────────────────────────────
#
# Provides Zygote-compatible alternatives for the standard boltz_forward() path.
# All functions are additive — no existing code is modified.
#
# Key differentiability fixes:
#   - randn!() mutation → randn() inside Zygote.@ignore
#   - push!() trajectory → overwrite atom_coords only (no trajectory)
#   - one_hot(Int.(msa)) → pre-one-hotted msa_onehot feature
#   - Int.(dist_idx) binning → Zygote.@ignore
#   - compute_ptms CPU loops → skipped (only logits needed for design)
#   - weighted_rigid_align SVD → Zygote.@ignore
#   - coordinate augmentation → skipped (mosaic also skips for design)

using Zygote

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _zeros_like(x, dims...)

Allocate a zero array on the same device as `x`.
Wrapped in Zygote.@ignore since these are constant padding (no gradient needed).
"""
_zeros_like(x::AbstractArray, T::Type, dims...) = Zygote.@ignore fill!(similar(x, T, dims...), zero(T))
_zeros_like(x::AbstractArray, dims...) = _zeros_like(x, eltype(x), dims...)

"""
    _ones_like(x, dims...)

Allocate a ones array on the same device as `x`.
Wrapped in Zygote.@ignore since these are constant masks (no gradient needed).
"""
_ones_like(x::AbstractArray, T::Type, dims...) = Zygote.@ignore fill!(similar(x, T, dims...), one(T))
_ones_like(x::AbstractArray, dims...) = _ones_like(x, eltype(x), dims...)

"""
    _dict_to_gpu(feats::Dict)

Transfer all array values in a feature dict to GPU (if CUDA is loaded).
Non-array values are passed through.
"""
function _dict_to_gpu(feats::Dict)
    out = Dict{String,Any}()
    for (k, v) in feats
        if v isa AbstractArray
            out[k] = Onion.Flux.gpu(v)
        else
            out[k] = v
        end
    end
    return out
end

# _dict_to_cpu is defined in api.jl — reuse that definition

# ─────────────────────────────────────────────────────────────────────────────
# set_binder_sequence: inject soft sequence into Boltz2 features
# ─────────────────────────────────────────────────────────────────────────────

"""
    set_binder_sequence(soft_seq, feats, binder_length)

Inject a soft (20, binder_length, 1) amino acid probability distribution into
Boltz2 features. Mirrors mosaic's `set_binder_sequence` (boltz2.py:197-227).

The 20 standard AAs are padded to 33 Boltz tokens:
  [2 zeros | 20 AAs | 11 zeros]  (positions 1-2 are pad/gap, 3-22 are AAs, 23-33 are non-canonical)

Modifies: res_type, msa_onehot (first MSA row for binder), profile
"""
function set_binder_sequence(soft_seq, feats, binder_length)
    # soft_seq: (20, binder_length, B)
    B = size(soft_seq, 3)

    # Pad 20 AAs → 33 Boltz tokens: [2 zeros | 20 AAs | 11 zeros]
    padded = cat(
        _zeros_like(soft_seq, Float32, 2, binder_length, B),
        soft_seq,
        _zeros_like(soft_seq, Float32, 11, binder_length, B);
        dims=1
    )  # (33, binder_length, B)

    # Build new feature dict (shallow copy — only replace what we change)
    new_feats = Dict{String,Any}()
    for (k, v) in feats
        new_feats[k] = v
    end

    # res_type: (num_tokens, T, B) — replace binder positions (1:binder_length)
    new_feats["res_type"] = _replace_binder_slice(feats["res_type"], padded, binder_length)

    # msa_onehot: (num_tokens, S, T, B) — replace first row of binder positions
    new_feats["msa_onehot"] = _replace_msa_first_row(feats["msa_onehot"], padded, binder_length)

    # profile: (num_tokens+1, T, B) — replace binder profile
    new_feats["profile"] = _replace_binder_profile(feats["profile"], padded, binder_length, feats)

    return new_feats
end

"""
    _replace_binder_slice(res_type, padded, binder_length)

Replace the first `binder_length` tokens in res_type with the soft padded sequence.
Uses differentiable masking instead of index assignment.
"""
function _replace_binder_slice(res_type, padded, binder_length)
    # res_type: (C, T, B), padded: (C, binder_length, B)
    C, T, B = size(res_type)

    # Create a mask: 1 for binder positions, 0 for target
    binder_mask = cat(
        _ones_like(padded, Float32, 1, binder_length, B),
        _zeros_like(padded, Float32, 1, T - binder_length, B);
        dims=2
    )  # (1, T, B)

    # Pad the soft sequence to full length T
    padded_full = cat(
        padded,
        _zeros_like(padded, Float32, C, T - binder_length, B);
        dims=2
    )  # (C, T, B)

    # Differentiable replacement: soft_seq where binder, original where target
    return padded_full .* binder_mask .+ res_type .* (1f0 .- binder_mask)
end

"""
    _replace_msa_first_row(msa_onehot, padded, binder_length)

Replace the first MSA row at binder positions with the soft padded sequence.
msa_onehot: (num_tokens, S, T, B) — S is the MSA depth.
"""
function _replace_msa_first_row(msa_onehot, padded, binder_length)
    C, S, T, B = size(msa_onehot)

    # Mask for binder positions in the first MSA row only
    binder_mask_4d = cat(
        _ones_like(padded, Float32, 1, 1, binder_length, B),
        _zeros_like(padded, Float32, 1, 1, T - binder_length, B);
        dims=3
    )  # (1, 1, T, B) — broadcasts over C and S

    # First-row mask: 1 for row 1 only
    row_mask = cat(
        _ones_like(padded, Float32, 1, 1, 1, B),
        _zeros_like(padded, Float32, 1, S - 1, 1, B);
        dims=2
    )  # (1, S, 1, B)

    combined_mask = binder_mask_4d .* row_mask  # (1, S, T, B)

    # Soft sequence reshaped for first row: (C, 1, binder_length, B)
    padded_row = reshape(padded, C, 1, binder_length, B)
    padded_full = cat(
        padded_row,
        _zeros_like(padded, Float32, C, 1, T - binder_length, B);
        dims=3
    )  # (C, 1, T, B)

    # Expand to full MSA depth
    padded_full_s = cat(
        padded_full,
        _zeros_like(padded, Float32, C, S - 1, T, B);
        dims=2
    )  # (C, S, T, B)

    return padded_full_s .* combined_mask .+ msa_onehot .* (1f0 .- combined_mask)
end

"""
    _replace_binder_profile(profile, padded, binder_length, feats)

Replace the binder portion of the profile feature.
profile: (num_tokens, T, B) — same shape as res_type (33 channels).
InputEmbedder concatenates deletion_mean on the fly when calling msa_profile_encoding.
"""
function _replace_binder_profile(profile, padded, binder_length, feats)
    C, T, B = size(profile)  # C = num_tokens = 33

    # padded is already (33, binder_length, B) — same shape as profile channels
    # Pad to full length T
    padded_full = cat(
        padded,
        _zeros_like(padded, Float32, C, T - binder_length, B);
        dims=2
    )  # (C, T, B)

    # Binder mask
    binder_mask = cat(
        _ones_like(padded, Float32, 1, binder_length, B),
        _zeros_like(padded, Float32, 1, T - binder_length, B);
        dims=2
    )  # (1, T, B)

    return padded_full .* binder_mask .+ profile .* (1f0 .- binder_mask)
end

# ─────────────────────────────────────────────────────────────────────────────
# _msa_forward_diff: differentiable MSAModule forward
# ─────────────────────────────────────────────────────────────────────────────

"""
    _msa_forward_diff(m::MSAModule, z, emb, feats; use_kernels=false)

Differentiable replacement for `MSAModule.__call__`. Instead of calling
`one_hot(Int.(feats["msa"]))`, reads pre-one-hotted `feats["msa_onehot"]`.

The msa_onehot should already be (num_tokens, S, T, B) with float values.
"""
function _msa_forward_diff(m::MSAModule, z, emb, feats; use_kernels::Bool=false)
    # Chunk size logic (same as standard path)
    if !Onion.bg_istraining()
        if size(z, 2) > chunk_size_threshold
            chunk_heads_pwa = true
            chunk_size_transition_z = 64
            chunk_size_transition_msa = 32
            chunk_size_outer_product = 4
            chunk_size_tri_attn = 128
        else
            chunk_heads_pwa = false
            chunk_size_transition_z = nothing
            chunk_size_transition_msa = nothing
            chunk_size_outer_product = nothing
            chunk_size_tri_attn = 512
        end
    else
        chunk_heads_pwa = false
        chunk_size_transition_z = nothing
        chunk_size_transition_msa = nothing
        chunk_size_outer_product = nothing
        chunk_size_tri_attn = nothing
    end

    # KEY DIFFERENCE: use pre-one-hotted MSA (differentiable w.r.t. soft sequence)
    # msa_onehot: (num_tokens, S, T, B) — already float, already permuted
    msa = feats["msa_onehot"]

    has_deletion = reshape(feats["has_deletion"], 1, size(feats["has_deletion"],1), size(feats["has_deletion"],2), size(feats["has_deletion"],3))
    deletion_value = reshape(feats["deletion_value"], 1, size(feats["deletion_value"],1), size(feats["deletion_value"],2), size(feats["deletion_value"],3))
    is_paired = reshape(feats["msa_paired"], 1, size(feats["msa_paired"],1), size(feats["msa_paired"],2), size(feats["msa_paired"],3))
    msa_mask = feats["msa_mask"]
    token_mask = feats["token_pad_mask"]
    token_mask = token_mask .* reshape(token_mask, 1, size(token_mask,1), size(token_mask,2))

    if m.use_paired_feature
        m_feat = cat(msa, has_deletion, deletion_value, is_paired; dims=1)
    else
        m_feat = cat(msa, has_deletion, deletion_value; dims=1)
    end

    m_feat = m.msa_proj(m_feat)
    m_feat = m_feat .+ reshape(m.s_proj(emb), size(m_feat,1), 1, size(m_feat,3), size(m_feat,4))

    for layer in m.layers
        z, m_feat = layer(z, m_feat, token_mask, msa_mask;
            chunk_heads_pwa=chunk_heads_pwa,
            chunk_size_transition_z=chunk_size_transition_z,
            chunk_size_transition_msa=chunk_size_transition_msa,
            chunk_size_outer_product=chunk_size_outer_product,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_kernels=use_kernels)
    end

    return z
end

# ─────────────────────────────────────────────────────────────────────────────
# Gradient checkpointing for diffusion steps
# ─────────────────────────────────────────────────────────────────────────────

"""
    _checkpointed_denoise(ad, atom_coords_noisy, t_hat, net_kwargs)

Gradient-checkpointed wrapper around `preconditioned_network_forward`.
Forward: runs the denoising network normally but discards activations.
Backward: recomputes forward from inputs, then computes gradient.
Costs 2x compute per step but reduces peak memory from O(num_steps) to O(1)
network activations.
"""
function _checkpointed_denoise(ad, atom_coords_noisy, t_hat, net_kwargs)
    return preconditioned_network_forward(ad, atom_coords_noisy, t_hat, net_kwargs; training=false)
end

Zygote.@adjoint function _checkpointed_denoise(ad, atom_coords_noisy, t_hat, net_kwargs)
    # Forward: run network, return result (Zygote would normally store activations here,
    # but our custom adjoint discards them by recomputing in the pullback)
    result = preconditioned_network_forward(ad, atom_coords_noisy, t_hat, net_kwargs; training=false)

    return result, function(Δ)
        # Backward: recompute forward to get pullback, then differentiate
        _, back = Zygote.pullback(
            (coords, kwargs) -> preconditioned_network_forward(ad, coords, t_hat, kwargs; training=false),
            atom_coords_noisy, net_kwargs,
        )
        g_coords, g_kwargs = back(Δ)
        return (nothing, g_coords, nothing, g_kwargs)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Gradient checkpointing for pairformer blocks
# ─────────────────────────────────────────────────────────────────────────────

"""
    _checkpointed_pairformer_layer(layer, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)

Gradient-checkpointed wrapper around a single PairformerLayer.
Forward: runs the layer normally but discards intermediate activations.
Backward: recomputes forward from (s, z) inputs, then computes gradient.
Costs 2x compute per block but reduces peak memory from O(num_blocks) to O(1)
per-block activations.
"""
function _checkpointed_pairformer_layer(layer, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
    return layer(s, z, mask, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)
end

Zygote.@adjoint function _checkpointed_pairformer_layer(layer, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
    result = layer(s, z, mask, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)

    return result, function(Δ)
        _, back = Zygote.pullback(
            (s_, z_) -> layer(s_, z_, mask, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels),
            s, z,
        )
        gs, gz = back(Δ)
        return (nothing, gs, gz, nothing, nothing, nothing, nothing)
    end
end

"""
    _checkpointed_pairformer_noseq_layer(layer, z, pair_mask, chunk_size_tri_attn, use_kernels)

Gradient-checkpointed wrapper for PairformerNoSeqLayer (used in confidence module).
"""
function _checkpointed_pairformer_noseq_layer(layer, z, pair_mask, chunk_size_tri_attn, use_kernels)
    return layer(z, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)
end

Zygote.@adjoint function _checkpointed_pairformer_noseq_layer(layer, z, pair_mask, chunk_size_tri_attn, use_kernels)
    result = layer(z, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)

    return result, function(Δ)
        _, back = Zygote.pullback(
            z_ -> layer(z_, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels),
            z,
        )
        gz = back(Δ)[1]
        return (nothing, gz, nothing, nothing, nothing)
    end
end

"""
    _checkpointed_pairformer(pf::PairformerModule, s, z, mask, pair_mask; use_kernels=false)

Run a PairformerModule with per-block gradient checkpointing.
Each block's intermediate activations are discarded after the forward pass and
recomputed during backward. Reduces memory from O(num_blocks) to O(1) per-block
activations at the cost of 2x compute.
"""
function _checkpointed_pairformer(pf, s, z, mask, pair_mask; use_kernels::Bool=false)
    chunk_size_tri_attn = size(z, 2) > 512 ? 128 : 512

    for layer in pf.layers
        s, z = _checkpointed_pairformer_layer(layer, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
    end
    return s, z
end

"""
    _checkpointed_pairformer_noseq(pf, z, pair_mask; use_kernels=false)

Run a PairformerNoSeqModule or PairformerModule (no-seq variant) with per-block
gradient checkpointing.
"""
function _checkpointed_pairformer_noseq(pf, z, pair_mask; use_kernels::Bool=false)
    chunk_size_tri_attn = size(z, 2) > 512 ? 128 : 512

    for layer in pf.layers
        z = _checkpointed_pairformer_noseq_layer(layer, z, pair_mask, chunk_size_tri_attn, use_kernels)
    end
    return z
end

# ─────────────────────────────────────────────────────────────────────────────
# sample_differentiable: non-mutating diffusion sampling
# ─────────────────────────────────────────────────────────────────────────────

"""
    sample_differentiable(ad::AtomDiffusion; atom_mask, key=nothing, ...)

Non-mutating diffusion sampling compatible with Zygote AD.

Key differences from `sample()`:
- `randn()` instead of `randn!()`, wrapped in `Zygote.@ignore`
- No `push!()` trajectory tracking — only final coords returned
- No coordinate augmentation (skipped for design, mosaic does too)
- `weighted_rigid_align` wrapped in `Zygote.@ignore` when used
"""
function sample_differentiable(
    ad::AtomDiffusion;
    atom_mask,
    num_sampling_steps=nothing,
    multiplicity::Int=1,
    step_scale=nothing,
    noise_scale=nothing,
    sampling_schedule=nothing,
    time_dilation=nothing,
    time_dilation_start=nothing,
    time_dilation_end=nothing,
    atom_coords_init=nothing,
    key=nothing,
    network_condition_kwargs...
)
    num_sampling_steps = default(num_sampling_steps, ad.num_sampling_steps)

    # Step scale schedule (same logic as sample())
    if ad.step_scale_function == "beta"
        step_scales = beta_step_scale_schedule(ad, num_sampling_steps)
    else
        ss = default(step_scale, ad.step_scale)
        ss === nothing && error("step_scale is nothing")
        step_scales = fill(Float32(ss), num_sampling_steps)
    end

    # Noise scale schedule
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
    shape = (3, size(atom_mask_rep, 1), size(atom_mask_rep, 2))

    # Sigma schedule
    eff_schedule = default(sampling_schedule, ad.sampling_schedule)
    if eff_schedule == "af3"
        sigmas = sample_schedule_af3(ad, num_sampling_steps)
    elseif eff_schedule == "dilated"
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

    # Initialize with fixed random noise (outside AD graph)
    if atom_coords_init === nothing
        init_noise = Zygote.@ignore begin
            rng = key !== nothing ? Random.MersenneTwister(key) : Random.default_rng()
            noise = randn(rng, Float32, shape)
            copyto!(similar(atom_mask_rep, Float32, shape), noise)
        end
        atom_coords = init_sigma .* init_noise
    else
        atom_coords = Float32.(atom_coords_init)
    end

    # Precompute all per-step noise and alignment targets (outside AD graph).
    # These are cached so that gradient-checkpointed recomputation is exact.
    cached_eps = Zygote.@ignore begin
        eps_list = Vector{Any}(undef, num_sampling_steps)
        for step_idx in 1:num_sampling_steps
            sigma_tm = sigmas[step_idx]
            gamma = gammas[step_idx + 1]
            t_hat = sigma_tm * (1 + gamma)
            noise_var = noise_scales[step_idx]^2 * (t_hat^2 - sigma_tm^2)
            rng_step = key !== nothing ? Random.MersenneTwister(hash(key + step_idx)) : Random.default_rng()
            step_noise = randn(rng_step, Float32, shape)
            step_noise_dev = copyto!(similar(atom_mask_rep, Float32, shape), step_noise)
            eps_list[step_idx] = noise_scales[step_idx] * sqrt(noise_var) .* step_noise_dev
        end
        eps_list
    end

    # Build the network kwargs dict once (not per-step)
    net_kwargs = Dict(
        "s_inputs" => network_condition_kwargs[:s_inputs],
        "s_trunk" => network_condition_kwargs[:s_trunk],
        "feats" => network_condition_kwargs[:feats],
        "multiplicity" => multiplicity,
        "diffusion_conditioning" => network_condition_kwargs[:diffusion_conditioning],
    )

    for step_idx in 1:num_sampling_steps
        sigma_tm = sigmas[step_idx]
        sigma_t = sigmas[step_idx + 1]
        gamma = gammas[step_idx + 1]
        t_hat = sigma_tm * (1 + gamma)

        # Center coordinates (differentiable)
        atom_coords = center(atom_coords, atom_mask_rep)

        # Add cached noise (noise itself has no gradient)
        eps = Zygote.@ignore cached_eps[step_idx]
        atom_coords_noisy = atom_coords .+ eps

        # Denoising via score model — checkpointed to save memory.
        # Forward: run network, discard activations.
        # Backward: recompute forward from (atom_coords_noisy, t_hat, net_kwargs),
        # then compute gradient. Costs 2x compute but ~1/num_steps memory.
        atom_coords_denoised, net_out = _checkpointed_denoise(
            ad, atom_coords_noisy, t_hat, net_kwargs,
        )

        # Alignment: @ignore the SVD
        if ad.alignment_reverse_diff
            atom_coords_noisy = Zygote.@ignore begin
                weighted_rigid_align(
                    atom_coords_noisy,
                    atom_coords_denoised,
                    Float32.(atom_mask_rep),
                    atom_mask_rep .> 0.5f0,
                )
            end
        end

        # Denoising step (differentiable)
        denoised_over_sigma = (atom_coords_noisy .- atom_coords_denoised) ./ t_hat
        atom_coords = atom_coords_noisy .+ step_scales[step_idx] .* (sigma_t - t_hat) .* denoised_over_sigma
    end

    return Dict{String,Any}(
        "sample_atom_coords" => atom_coords,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# _confidence_forward_gpu: GPU-compatible confidence forward
# ─────────────────────────────────────────────────────────────────────────────

"""
    _confidence_forward_gpu(cm::ConfidenceModule, s_inputs, s, z, x_pred, feats, pred_distogram_logits; multiplicity=1, use_kernels=false)

GPU-compatible confidence forward that produces pLDDT and PAE logits without
CPU transfers. Skips `compute_ptms` and complex metric aggregation.

Key fixes for differentiability:
- Distance binning `Int.(sum(d .> boundaries))` → `Zygote.@ignore`
- `Matrix{Float32}(I, N, N)` → `Zygote.@ignore`
- No `compute_ptms` (not needed for design losses)
- Returns: plddt_logits, pae_logits, pde_logits, plddt, pae, complex_plddt, interaction_pae
"""
function _confidence_forward_gpu(
    cm::ConfidenceModule,
    s_inputs,
    s,
    z,
    x_pred,
    feats,
    pred_distogram_logits;
    multiplicity::Int=1,
    use_kernels::Bool=false,
)
    # Normalize inputs (differentiable)
    s_inputs_n = cm.s_inputs_norm(s_inputs)
    if !cm.no_update_s && cm.s_norm !== nothing
        s = cm.s_norm(s)
    end
    if cm.add_s_input_to_s && cm.s_input_to_s !== nothing
        s = s .+ cm.s_input_to_s(s_inputs_n)
    end
    z = cm.z_norm(z)

    s = _maybe_repeat_interleave(s, multiplicity)

    token_z = cm.token_z
    N = size(s_inputs_n, 2)
    B = size(s_inputs_n, 3)

    # Pairwise from single (differentiable)
    z = z .+ reshape(cm.s_to_z(s_inputs_n), token_z, N, 1, B) .+
              reshape(cm.s_to_z_transpose(s_inputs_n), token_z, 1, N, B)

    if cm.add_s_to_z_prod && cm.s_to_z_prod_out !== nothing
        prod = reshape(cm.s_to_z_prod_in1(s_inputs_n), token_z, N, 1, B) .*
               reshape(cm.s_to_z_prod_in2(s_inputs_n), token_z, 1, N, B)
        z = z .+ cm.s_to_z_prod_out(prod)
    end

    z = _maybe_repeat_interleave(z, multiplicity)
    s_inputs_rep = _maybe_repeat_interleave(s_inputs_n, multiplicity)

    # Compute representative atom coordinates (differentiable)
    token_to_rep_atom_ff = repeat_interleave_batch(feats["token_to_rep_atom"], multiplicity)

    x_pred_ff = x_pred
    if ndims(x_pred_ff) == 4
        x_pred_ff = reshape(x_pred_ff, size(x_pred_ff,1), size(x_pred_ff,2), size(x_pred_ff,3)*size(x_pred_ff,4))
    end

    x_pred_batched = permutedims(x_pred_ff, (2, 1, 3))
    x_pred_repr = NNlib.batched_mul(token_to_rep_atom_ff, x_pred_batched)
    x_pred_repr_ff = permutedims(x_pred_repr, (2, 1, 3))

    # Pairwise distances (differentiable)
    d = pairwise_distance_batch(permutedims(x_pred_repr_ff, (3, 2, 1)))

    # Distance binning — NOT differentiable, wrap in @ignore
    # Gradient flows through trunk z, not through distance bins
    distogram = Zygote.@ignore begin
        dist_idx = sum(reshape(d, size(d,1), size(d,2), size(d,3), 1) .>
                       reshape(cm.boundaries, 1, 1, 1, :); dims=4)
        dist_idx_int = Int.(dropdims(dist_idx; dims=4))
        dist_idx_ff = permutedims(dist_idx_int, (2, 3, 1))
        cm.dist_bin_pairwise_embed(dist_idx_ff)
    end
    z = z .+ distogram

    # Optional z_input recomputation — all fixed-feature encodings, @ignore
    if cm.add_z_input_to_z === true && cm.rel_pos !== nothing
        z_input = Zygote.@ignore begin
            zi = cm.rel_pos(feats)
            zi = zi .+ cm.token_bonds(feats["token_bonds"])
            if cm.bond_type_feature && cm.token_bonds_type !== nothing
                zi = zi .+ cm.token_bonds_type(feats["type_bonds"])
            end
            zi = zi .+ cm.contact_conditioning(feats)
            zi
        end
        z = z .+ _maybe_repeat_interleave(z_input, multiplicity)
    end

    # Confidence pairformer (differentiable — runs on GPU)
    mask = _maybe_repeat_interleave(feats["token_pad_mask"], multiplicity)
    pair_mask = reshape(mask, size(mask,1), 1, size(mask,2)) .* reshape(mask, 1, size(mask,1), size(mask,2))

    s, z = _checkpointed_pairformer(cm.pairformer_stack, s, z, mask, pair_mask; use_kernels=use_kernels)

    # ── Confidence heads (logits only, skip ptm/iptm) ──
    ch = cm.confidence_heads

    Bm = size(s, 3)

    # Logit projections (all differentiable linear ops)
    # Chain masks are fixed (not differentiable) — @ignore to prevent CPU/GPU mismatch in backward
    same_chain_mask, diff_chain_mask = Zygote.@ignore begin
        asym_id_ff = repeat_interleave_batch(feats["asym_id"], multiplicity)
        is_same = _unsqueeze_local(asym_id_ff, 2) .== _unsqueeze_local(asym_id_ff, 1)
        is_diff = .!is_same
        same_f32 = reshape(Float32.(is_same), 1, size(is_same,1), size(is_same,2), size(is_same,3))
        diff_f32 = reshape(Float32.(is_diff), 1, size(is_diff,1), size(is_diff,2), size(is_diff,3))
        (same_f32, diff_f32)
    end

    pae_logits = if ch.use_separate_heads
        pae_intra = ch.to_pae_intra_logits(z) .* same_chain_mask
        pae_inter = ch.to_pae_inter_logits(z) .* diff_chain_mask
        pae_intra .+ pae_inter
    else
        ch.to_pae_logits(z)
    end

    pde_logits = if ch.use_separate_heads
        z_sym = z .+ permutedims(z, (1, 3, 2, 4))
        pde_intra = ch.to_pde_intra_logits(z_sym) .* same_chain_mask
        pde_inter = ch.to_pde_inter_logits(z_sym) .* diff_chain_mask
        pde_intra .+ pde_inter
    else
        ch.to_pde_logits(z .+ permutedims(z, (1, 3, 2, 4)))
    end

    plddt_logits = ch.to_plddt_logits(s)
    resolved_logits = ch.to_resolved_logits(s)

    # Aggregated metrics (compute_aggregated_metric uses softmax — GPU-safe)
    pae_logits_py = _to_py_logits4(pae_logits)
    pde_logits_py = _to_py_logits4(pde_logits)

    # plddt aggregation
    plddt_logits_py = _to_py_logits3(plddt_logits)
    plddt = Onion.compute_aggregated_metric(plddt_logits_py)

    pae = Onion.compute_aggregated_metric(pae_logits_py; end_value=32f0)

    # All feature-derived masks wrapped in @ignore to prevent CPU/GPU mismatch in backward
    token_pad_mask_py, target_designchain_inter_mask, iplddt_weight = Zygote.@ignore begin
        token_pad_mask_ff = repeat_interleave_batch(feats["token_pad_mask"], multiplicity)
        _tpm = permutedims(token_pad_mask_ff, (2, 1))

        design_mask_ff = repeat_interleave_batch(feats["design_mask"], multiplicity)
        design_mask_py = permutedims(design_mask_ff, (2, 1))
        chain_design_mask_ff = repeat_interleave_batch(feats["chain_design_mask"], multiplicity)
        chain_design_mask_py = permutedims(chain_design_mask_ff, (2, 1))
        asym_id_ff_py = permutedims(repeat_interleave_batch(feats["asym_id"], multiplicity), (2, 1))

        eye_cpu = reshape(1f0 .- Matrix{Float32}(I, N, N), 1, N, N)
        eye_mask_dev = copyto!(similar(_tpm, Float32, size(eye_cpu)), eye_cpu)

        token_pad_pair_mask = _unsqueeze_local(_tpm, 3) .* _unsqueeze_local(_tpm, 2) .* eye_mask_dev

        is_chain_design_token = if sum(feats["design_mask"]) > 0
            chain_design_mask_py
        else
            design_mask_py
        end
        is_target_token = if sum(feats["design_mask"]) > 0
            1f0 .- chain_design_mask_py
        else
            design_mask_py
        end

        _tdim = token_pad_pair_mask .* (
            _unsqueeze_local(is_chain_design_token, 3) .* _unsqueeze_local(is_target_token, 2) .+
            _unsqueeze_local(is_target_token, 3) .* _unsqueeze_local(is_chain_design_token, 2)
        )

        # ipLDDT weight computation
        token_type_ff = repeat_interleave_batch(feats["mol_type"], multiplicity)
        token_type_py = permutedims(token_type_ff, (2, 1))
        is_protein_token = Float32.(token_type_py .== chain_type_ids["PROTEIN"])
        is_ligand_token = Float32.(token_type_py .== chain_type_ids["NONPOLYMER"])

        is_contact = Float32.(d .< 8f0)
        is_different_chain = Float32.(_unsqueeze_local(asym_id_ff_py, 3) .!= _unsqueeze_local(asym_id_ff_py, 2))
        token_interface_mask = begin
            tim = maximum(is_contact .* is_different_chain .* reshape(1f0 .- is_ligand_token, size(is_ligand_token,1), size(is_ligand_token,2), 1); dims=3)
            dropdims(tim; dims=3)
        end
        token_non_interface_mask = (1f0 .- token_interface_mask) .* (1f0 .- is_ligand_token)

        _ipw = is_ligand_token .* 20f0 .+ token_interface_mask .* 10f0 .+ token_non_interface_mask .* 1f0

        (_tpm, _tdim, _ipw)
    end

    # Complex pLDDT (gradient flows through plddt only, masks are @ignored)
    complex_plddt = sum(plddt .* token_pad_mask_py; dims=2) ./ sum(token_pad_mask_py; dims=2)
    complex_plddt = dropdims(complex_plddt; dims=2)

    # Interaction PAE (gradient flows through pae only)
    interaction_pae = sum(pae .* target_designchain_inter_mask; dims=(2, 3)) ./
        (sum(target_designchain_inter_mask; dims=(2, 3)) .+ 1f-5)
    interaction_pae = dropdims(interaction_pae; dims=(2, 3))

    # ipLDDT using distogram contact probabilities
    pred_dist_py = _to_py_logits4(pred_distogram_logits)
    pred_dist_prob = NNlib.softmax(pred_dist_py; dims=4)
    pred_dist_prob = repeat_interleave_dim(pred_dist_prob, multiplicity, 1)

    # Contact probability from distogram (first 20 bins = < ~8A)
    num_dist_bins = size(pred_dist_prob, 4)
    contact_bins = min(20, num_dist_bins)
    prob_contact = sum(pred_dist_prob[:, :, :, 1:contact_bins]; dims=4)
    prob_contact = dropdims(prob_contact; dims=4)

    complex_iplddt = sum(plddt .* token_pad_mask_py .* iplddt_weight; dims=2) ./ sum(token_pad_mask_py .* iplddt_weight; dims=2)
    complex_iplddt = dropdims(complex_iplddt; dims=2)

    plddt_ff = permutedims(plddt, (2, 1))

    return Dict{String,Any}(
        "plddt_logits" => plddt_logits,
        "pae_logits" => pae_logits,
        "pde_logits" => pde_logits,
        "resolved_logits" => resolved_logits,
        "plddt" => plddt_ff,
        "pae" => permutedims(pae, (2, 3, 1)),
        "complex_plddt" => complex_plddt,
        "complex_iplddt" => complex_iplddt,
        "interaction_pae" => interaction_pae,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# boltz2_differentiable_forward: main entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    boltz2_differentiable_forward(model::BoltzModel, feats; recycling_steps=0, num_sampling_steps=nothing, multiplicity=1, key=nothing, step_scale=nothing, noise_scale=nothing, sampling_schedule=nothing, time_dilation=nothing, use_kernels=false)

Zygote-compatible Boltz2 forward pass. Reuses model weights but calls
differentiable replacements for MSA, diffusion sampling, and confidence.

Returns Dict with:
- "pdistogram": (num_bins, T, T, 1, B) distogram logits
- "sample_atom_coords": (3, M, B*mult) predicted atom coordinates
- "plddt_logits", "pae_logits", "pde_logits": confidence logits
- "plddt", "pae", "complex_plddt", "complex_iplddt", "interaction_pae": aggregated metrics
"""
function boltz2_differentiable_forward(
    model::BoltzModel,
    feats;
    recycling_steps::Int=0,
    num_sampling_steps=nothing,
    multiplicity::Int=1,
    key=nothing,
    step_scale=nothing,
    noise_scale=nothing,
    sampling_schedule=nothing,
    time_dilation=nothing,
    time_dilation_start=nothing,
    time_dilation_end=nothing,
    use_kernels::Bool=false,
)
    # 1. Input embedding — works as-is with soft res_type and profile
    s_inputs = model.input_embedder(feats)

    # 2. Pairwise initialization (same as boltz_forward)
    s_init = model.s_init(s_inputs)
    z_init = reshape(model.z_init_1(s_inputs), model.token_z, size(s_inputs,2), 1, size(s_inputs,3)) .+
        reshape(model.z_init_2(s_inputs), model.token_z, 1, size(s_inputs,2), size(s_inputs,3))

    # Fixed-feature encodings: @ignore because they don't depend on soft_seq
    # (operate on integer features like asym_id, residue_index, token_bonds, etc.)
    relative_position_encoding = Zygote.@ignore model.rel_pos(feats)
    z_init = z_init .+ relative_position_encoding
    z_init = z_init .+ Zygote.@ignore(model.token_bonds(feats["token_bonds"]))
    if model.bond_type_feature && model.token_bonds_type !== nothing
        z_init = z_init .+ Zygote.@ignore(model.token_bonds_type(feats["type_bonds"]))
    end
    z_init = z_init .+ Zygote.@ignore(model.contact_conditioning(feats))

    s = Onion.zeros_like(s_init)
    z = Onion.zeros_like(z_init)

    mask = feats["token_pad_mask"]
    pair_mask = reshape(mask, size(mask,1), 1, size(mask,2)) .* reshape(mask, 1, size(mask,1), size(mask,2))

    # 3. Recycling with @ignore between iterations
    for i in 1:(recycling_steps + 1)
        s = s_init .+ model.s_recycle(model.s_norm(s))
        z = z_init .+ model.z_recycle(model.z_norm(z))

        if model.use_token_distances && model.token_distance_module !== nothing
            z = z .+ Zygote.@ignore(model.token_distance_module(z, feats, pair_mask, relative_position_encoding; use_kernels=use_kernels))
        end
        if model.use_templates && model.template_module !== nothing
            z = z .+ Zygote.@ignore(model.template_module(z, feats, pair_mask; use_kernels=use_kernels))
        end

        # Differentiable MSA forward (uses pre-one-hotted msa_onehot)
        z = z .+ _msa_forward_diff(model.msa_module, z, s_inputs, feats; use_kernels=use_kernels)

        s, z = _checkpointed_pairformer(model.pairformer_module, s, z, mask, pair_mask; use_kernels=use_kernels)

        # Stop gradient between recycling iterations
        if i < recycling_steps + 1
            s = Zygote.@ignore copy(s)
            z = Zygote.@ignore copy(z)
        end
    end

    # 4. Distogram (already differentiable)
    pdistogram = model.distogram_module(z)

    # 5. Diffusion conditioning (already differentiable)
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

    # 6. Differentiable sampling (no mutation, no trajectory)
    struct_out = sample_differentiable(model.structure_module;
        s_trunk=s,
        s_inputs=s_inputs,
        feats=feats,
        atom_mask=feats["atom_pad_mask"],
        num_sampling_steps=num_sampling_steps,
        multiplicity=multiplicity,
        diffusion_conditioning=diffusion_conditioning,
        step_scale=step_scale,
        noise_scale=noise_scale,
        sampling_schedule=sampling_schedule,
        time_dilation=time_dilation,
        time_dilation_start=time_dilation_start,
        time_dilation_end=time_dilation_end,
        key=key,
    )

    # 7. Confidence on GPU (differentiable)
    out = Dict{String,Any}(
        "pdistogram" => pdistogram,
    )
    for (k, v) in struct_out
        out[k] = v
    end

    if model.confidence_prediction && model.confidence_module !== nothing
        conf_out = _confidence_forward_gpu(
            model.confidence_module,
            s_inputs,
            s,
            z,
            struct_out["sample_atom_coords"],
            feats,
            dropdims(pdistogram; dims=4);
            multiplicity=multiplicity,
            use_kernels=use_kernels,
        )
        for (k, v) in conf_out
            out[k] = v
        end
    end

    return out
end
