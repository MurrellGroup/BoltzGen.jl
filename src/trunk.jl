using Onion
using NNlib
import Onion: rearrange, @einops_str

const BGLayerNorm = Onion.BGLayerNorm

function pairwise_distance_batch(coords)
    # coords: (B, N, C)
    B, N, C = size(coords)
    diff = reshape(coords, B, N, 1, C) .- reshape(coords, B, 1, N, C)
    d = sqrt.(sum(diff .* diff; dims=4))
    return dropdims(d; dims=4)
end

@concrete struct BGEmbedding <: Onion.Layer
    weight
end

@layer BGEmbedding

function BGEmbedding(num_embeddings::Int, dim::Int; init::Symbol=:zeros)
    weight = zeros(Float32, dim, num_embeddings)
    if init === :torch
        torch_embedding_init!(weight)
    end
    return BGEmbedding(weight)
end

function (e::BGEmbedding)(idx)
    flat = vec(idx)
    out = e.weight[:, flat .+ 1]
    return reshape(out, size(e.weight,1), size(idx)...)
end

@concrete struct ContactConditioning <: Onion.Layer
    fourier_embedding
    encoder
    encoding_unspecified
    encoding_unselected
    cutoff_min
    cutoff_max
end

@layer ContactConditioning

function ContactConditioning(; token_z::Int, cutoff_min::Float32, cutoff_max::Float32)
    fourier_embedding = FourierEmbedding(token_z)
    encoder = BGLinear(token_z + length(contact_conditioning_info) - 1, token_z; bias=true, init=:default)
    Onion.torch_linear_init!(encoder.weight, encoder.bias)
    encoding_unspecified = zeros(Float32, token_z)
    encoding_unselected = zeros(Float32, token_z)
    return ContactConditioning(fourier_embedding, encoder, encoding_unspecified, encoding_unselected, cutoff_min, cutoff_max)
end

function (cc::ContactConditioning)(feats)
    contact_conditioning = feats["contact_conditioning"]
    contact_threshold = feats["contact_threshold"]

    cc_b = permutedims(contact_conditioning, (4,2,3,1))
    thresh_b = permutedims(contact_threshold, (3,1,2))

    thresh_norm = (thresh_b .- cc.cutoff_min) ./ (cc.cutoff_max - cc.cutoff_min)
    flat = reshape(thresh_norm, :)
    fourier = cc.fourier_embedding(flat)
    fourier = reshape(fourier, size(fourier,1), size(thresh_norm,1), size(thresh_norm,2), size(thresh_norm,3))
    fourier = permutedims(fourier, (2,3,4,1))

    cc_main = cc_b[:, :, :, 3:end]
    cc_cat = cat(cc_main, reshape(thresh_norm, size(thresh_norm,1), size(thresh_norm,2), size(thresh_norm,3), 1), fourier; dims=4)
    cc_cat = permutedims(cc_cat, (4,2,3,1))
    cc_enc = cc.encoder(cc_cat)
    cc_enc = permutedims(cc_enc, (4,2,3,1))

    mask_unspec = cc_b[:, :, :, 1:1]
    mask_unselect = cc_b[:, :, :, 2:2]
    mask_spec = 1f0 .- (mask_unspec .+ mask_unselect)

    unspec = reshape(cc.encoding_unspecified, 1, 1, 1, length(cc.encoding_unspecified)) .* mask_unspec
    unselect = reshape(cc.encoding_unselected, 1, 1, 1, length(cc.encoding_unselected)) .* mask_unselect

    out = cc_enc .* mask_spec .+ unspec .+ unselect
    return permutedims(out, (4,2,3,1))
end

@concrete struct InputEmbedder <: Onion.Layer
    token_s::Int
    add_method_conditioning::Bool
    add_modified_flag::Bool
    add_cyclic_flag::Bool
    add_mol_type_feat::Bool
    add_ph_flag::Bool
    add_temp_flag::Bool
    add_design_mask_flag::Bool
    add_binding_specification::Bool
    add_ss_specification::Bool
    atom_encoder
    atom_enc_proj_z
    atom_attention_encoder
    res_type_encoding
    msa_profile_encoding
    method_conditioning_init
    modified_conditioning_init
    cyclic_conditioning_init
    mol_type_conditioning_init
    ph_conditioning_init
    temp_conditioning_init
    binding_specification_conditioning_init
    design_mask_conditioning_init
    ss_specification_init
end

@layer InputEmbedder

function InputEmbedder(
    atom_s::Int,
    atom_z::Int,
    token_s::Int,
    token_z::Int,
    atoms_per_window_queries::Int,
    atoms_per_window_keys::Int,
    atom_feature_dim::Int;
    atom_encoder_depth::Int=3,
    atom_encoder_heads::Int=4,
    activation_checkpointing::Bool=false,
    add_method_conditioning::Bool=false,
    add_modified_flag::Bool=false,
    add_cyclic_flag::Bool=false,
    add_mol_type_feat::Bool=false,
    add_ph_flag::Bool=false,
    add_temp_flag::Bool=false,
    add_design_mask_flag::Bool=false,
    add_binding_specification::Bool=false,
    add_ss_specification::Bool=false,
)
    atom_encoder = AtomEncoder(
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim;
        structure_prediction=false,
    )

    atom_enc_proj_z = (BGLayerNorm(atom_z; eps=1f-5), LinearNoBias(atom_z, atom_encoder_depth * atom_encoder_heads))
    Onion.torch_linear_init!(atom_enc_proj_z[2].weight)

    atom_attention_encoder = AtomAttentionEncoder(
        atom_s,
        token_s,
        atoms_per_window_queries,
        atoms_per_window_keys;
        atom_encoder_depth=atom_encoder_depth,
        atom_encoder_heads=atom_encoder_heads,
        structure_prediction=false,
        activation_checkpointing=activation_checkpointing,
        tfmr_s=token_s,
    )

    res_type_encoding = LinearNoBias(num_tokens, token_s)
    msa_profile_encoding = LinearNoBias(num_tokens + 1, token_s)
    Onion.torch_linear_init!(res_type_encoding.weight)
    Onion.torch_linear_init!(msa_profile_encoding.weight)

    method_conditioning_init = add_method_conditioning ? BGEmbedding(num_method_types, token_s) : nothing
    modified_conditioning_init = add_modified_flag ? BGEmbedding(2, token_s) : nothing
    cyclic_conditioning_init = add_cyclic_flag ? BGLinear(1, token_s; bias=false, init=:default) : nothing
    mol_type_conditioning_init = add_mol_type_feat ? BGEmbedding(length(chain_type_ids), token_s) : nothing
    ph_conditioning_init = add_ph_flag ? BGEmbedding(num_ph_bins, token_s) : nothing
    temp_conditioning_init = add_temp_flag ? BGEmbedding(num_temp_bins, token_s) : nothing
    binding_specification_conditioning_init = add_binding_specification ? BGEmbedding(length(binding_types), token_s) : nothing
    design_mask_conditioning_init = add_design_mask_flag ? BGEmbedding(2, token_s) : nothing
    ss_specification_init = add_ss_specification ? BGEmbedding(length(ss_types), token_s) : nothing

    if cyclic_conditioning_init !== nothing
        cyclic_conditioning_init.weight .= 0f0
    end

    return InputEmbedder(
        token_s,
        add_method_conditioning,
        add_modified_flag,
        add_cyclic_flag,
        add_mol_type_feat,
        add_ph_flag,
        add_temp_flag,
        add_design_mask_flag,
        add_binding_specification,
        add_ss_specification,
        atom_encoder,
        atom_enc_proj_z,
        atom_attention_encoder,
        res_type_encoding,
        msa_profile_encoding,
        method_conditioning_init,
        modified_conditioning_init,
        cyclic_conditioning_init,
        mol_type_conditioning_init,
        ph_conditioning_init,
        temp_conditioning_init,
        binding_specification_conditioning_init,
        design_mask_conditioning_init,
        ss_specification_init,
    )
end

function (ie::InputEmbedder)(feats; affinity::Bool=false)
    res_type = feats["res_type"]
    if affinity
        profile = feats["profile_affinity"]
        deletion_mean = feats["deletion_mean_affinity"]
    else
        profile = feats["profile"]
        deletion_mean = feats["deletion_mean"]
    end

    q, c, p, to_keys = ie.atom_encoder(feats)
    ln_p, lin_p = ie.atom_enc_proj_z
    atom_enc_bias = lin_p(ln_p(p))

    a, _, _, _ = ie.atom_attention_encoder(
        feats=feats,
        q=q,
        c=c,
        atom_enc_bias=atom_enc_bias,
        to_keys=to_keys,
    )

    s = a .+ ie.res_type_encoding(res_type)
    s = s .+ ie.msa_profile_encoding(cat(profile, reshape(deletion_mean, 1, size(deletion_mean,1), size(deletion_mean,2)); dims=1))

    if ie.add_method_conditioning
        s = s .+ ie.method_conditioning_init(feats["method_feature"])
    end
    if ie.add_modified_flag
        s = s .+ ie.modified_conditioning_init(feats["modified"])
    end
    if ie.add_cyclic_flag
        cyclic = clamp.(feats["cyclic"], 0f0, 1f0)
        cyclic = reshape(cyclic, 1, size(cyclic,1), size(cyclic,2))
        s = s .+ ie.cyclic_conditioning_init(cyclic)
    end
    if ie.add_mol_type_feat
        s = s .+ ie.mol_type_conditioning_init(feats["mol_type"])
    end
    if ie.add_ph_flag
        s = s .+ ie.ph_conditioning_init(feats["ph_feature"])
    end
    if ie.add_temp_flag
        s = s .+ ie.temp_conditioning_init(feats["temp_feature"])
    end
    if ie.add_design_mask_flag
        s = s .+ ie.design_mask_conditioning_init(feats["design_mask"])
    end
    if ie.add_binding_specification
        s = s .+ ie.binding_specification_conditioning_init(feats["binding_type"])
    end
    if ie.add_ss_specification
        s = s .+ ie.ss_specification_init(feats["ss_type"])
    end

    return s
end

@concrete struct TemplateModule <: Onion.Layer
    min_dist
    max_dist
    num_bins
    z_norm
    v_norm
    z_proj
    a_proj
    u_proj
    pairformer
end

@layer TemplateModule

function TemplateModule(
    token_z::Int,
    template_dim::Int,
    template_blocks::Int;
    dropout::Float32=0.25f0,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    post_layer_norm::Bool=false,
    activation_checkpointing::Bool=false,
    min_dist::Float32=3.25f0,
    max_dist::Float32=50.75f0,
    num_bins::Int=38,
    miniformer_blocks::Bool=false,
)
    z_norm = BGLayerNorm(token_z; eps=1f-5)
    v_norm = BGLayerNorm(template_dim; eps=1f-5)
    z_proj = LinearNoBias(token_z, template_dim)
    a_proj = LinearNoBias(num_tokens * 2 + num_bins + 5, template_dim)
    u_proj = LinearNoBias(template_dim, token_z)
    Onion.torch_linear_init!(z_proj.weight)
    Onion.torch_linear_init!(a_proj.weight)
    Onion.torch_linear_init!(u_proj.weight)

    pairformer = if miniformer_blocks
        Onion.MiniformerNoSeqModule(template_dim, template_blocks; dropout=dropout, post_layer_norm=post_layer_norm, activation_checkpointing=activation_checkpointing)
    else
        Onion.PairformerNoSeqModule(template_dim, template_blocks; dropout=dropout, pairwise_head_width=pairwise_head_width, pairwise_num_heads=pairwise_num_heads, post_layer_norm=post_layer_norm, activation_checkpointing=activation_checkpointing)
    end

    return TemplateModule(min_dist, max_dist, num_bins, z_norm, v_norm, z_proj, a_proj, u_proj, pairformer)
end

function (tm::TemplateModule)(z, feats, pair_mask; use_kernels::Bool=false)
    # convert to python-like layout for clarity
    z_b = permutedims(z, (4,2,3,1)) # (B, N, N, Cz)
    asym_id = permutedims(feats["asym_id"], (2,1)) # (B, N)
    res_type = permutedims(feats["template_restype"], (4,3,2,1)) # (B, T, N, C)
    frame_rot = permutedims(feats["template_frame_rot"], (5,4,3,2,1)) # (B, T, N, 3, 3)
    frame_t = permutedims(feats["template_frame_t"], (4,3,2,1)) # (B, T, N, 3)
    frame_mask = permutedims(feats["template_mask_frame"], (3,2,1)) # (B, T, N)
    cb_coords = permutedims(feats["template_cb"], (4,3,2,1)) # (B, T, N, 3)
    ca_coords = permutedims(feats["template_ca"], (4,3,2,1)) # (B, T, N, 3)
    cb_mask = permutedims(feats["template_mask_cb"], (3,2,1)) # (B, T, N)
    template_mask = permutedims(feats["template_mask"], (3,2,1)) # (B, T, N)

    B = size(res_type, 1)
    T = size(res_type, 2)
    N = size(res_type, 3)

    template_mask_any = any(template_mask .> 0.5f0; dims=3)
    template_mask_any = dropdims(template_mask_any; dims=3) # (B, T)
    num_templates = sum(template_mask_any; dims=2)
    num_templates = max.(num_templates, 1f0)

    b_cb_mask = reshape(cb_mask, B, T, N, 1) .* reshape(cb_mask, B, T, 1, N)
    b_cb_mask = reshape(b_cb_mask, B, T, N, N, 1)
    b_frame_mask = reshape(frame_mask, B, T, N, 1) .* reshape(frame_mask, B, T, 1, N)
    b_frame_mask = reshape(b_frame_mask, B, T, N, N, 1)

    asym_mask = reshape(asym_id, B, N, 1) .== reshape(asym_id, B, 1, N)
    asym_mask = reshape(asym_mask, B, 1, N, N)
    asym_mask = repeat(asym_mask, 1, T, 1, 1)

    cb_flat = reshape(cb_coords, B * T, N, 3)
    dists_flat = pairwise_distance_batch(cb_flat)
    dists = reshape(dists_flat, B, T, N, N)
    boundaries_cpu = Float32.(collect(range(tm.min_dist, tm.max_dist; length=tm.num_bins-1)))
    boundaries_dev = copyto!(similar(dists, Float32, length(boundaries_cpu)), boundaries_cpu)
    distogram = sum(reshape(dists, B, T, N, N, 1) .> reshape(boundaries_dev, 1, 1, 1, 1, :); dims=5)
    distogram = dropdims(distogram; dims=5)
    distogram = one_hot(Int.(distogram), tm.num_bins)

    # Vectorized rotation: vector[b,t,i,j,k] = Σ_l frame_rot[b,t,j,l,k] * (ca_coords[b,t,i,l] - frame_t[b,t,j,l])
    diff = reshape(ca_coords, B, T, N, 1, 3) .- reshape(frame_t, B, T, 1, N, 3)  # (B, T, N_i, N_j, 3_l)
    diff_bat = rearrange(diff, einops"b t i j l -> i l (j b t)")     # (N_i, 3_l, batch)
    rot_bat = rearrange(frame_rot, einops"b t j l k -> l k (j b t)") # (3_l, 3_k, batch)
    result_bat = NNlib.batched_mul(diff_bat, rot_bat)                  # (N_i, 3_k, batch)
    vector = rearrange(result_bat, einops"i k (j b t) -> b t i j k"; j=N, b=B, t=T)
    norm = abs.(vector)
    unit_vector = ifelse.(norm .> 0f0, vector ./ norm, 0f0)

    a_tij = cat(distogram, b_cb_mask, unit_vector, b_frame_mask; dims=5)
    a_tij = a_tij .* reshape(asym_mask, B, T, N, N, 1)

    res_type_i = reshape(res_type, B, T, N, 1, size(res_type,4))
    res_type_j = reshape(res_type, B, T, 1, N, size(res_type,4))
    res_type_i = repeat(res_type_i, 1, 1, 1, N, 1)
    res_type_j = repeat(res_type_j, 1, 1, N, 1, 1)

    a_tij = cat(a_tij, res_type_i, res_type_j; dims=5)
    a_tij = tm.a_proj(permutedims(a_tij, (5,3,4,2,1)))
    a_tij = permutedims(a_tij, (5,4,2,3,1)) # (B, T, N, N, template_dim)

    pair_mask_b = permutedims(pair_mask, (3,1,2))
    pair_mask_t = repeat(pair_mask_b, 1, T, 1, 1)
    pair_mask_t = reshape(pair_mask_t, B * T, N, N)
    pair_mask_t = permutedims(pair_mask_t, (2,3,1))

    v = tm.z_proj(tm.z_norm(z))
    v = permutedims(v, (4,2,3,1)) # (B, N, N, template_dim)
    v = reshape(v, B, 1, N, N, size(v,4)) .+ a_tij
    v = reshape(v, B * T, N, N, size(v,5))
    v = permutedims(v, (4,2,3,1)) # feature-first
    v = v .+ tm.pairformer(v, pair_mask_t; use_kernels=use_kernels)
    v = tm.v_norm(v)
    v = permutedims(v, (4,2,3,1)) # (B*T, N, N, template_dim)
    v = reshape(v, B, T, N, N, size(v,4))

    template_mask_any = reshape(template_mask_any, B, T, 1, 1, 1)
    num_templates = reshape(num_templates, B, 1, 1, 1, 1)
    u = sum(v .* template_mask_any; dims=2) ./ num_templates
    u = dropdims(u; dims=2) # (B, N, N, template_dim)
    u = permutedims(u, (4,2,3,1))
    u = tm.u_proj(NNlib.relu.(u))
    return u
end

@concrete struct TokenDistanceModule <: Onion.Layer
    min_dist
    max_dist
    num_bins
    use_token_distance_feats::Bool
    z_norm
    v_norm
    z_proj
    a_proj
    u_proj
    pairformer
    token_distance_encoder
end

@layer TokenDistanceModule

function TokenDistanceModule(
    token_z::Int,
    token_distance_dim::Int,
    token_distance_blocks::Int;
    dropout::Float32=0.25f0,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    post_layer_norm::Bool=false,
    activation_checkpointing::Bool=false,
    min_dist::Float32=3.25f0,
    max_dist::Float32=50.75f0,
    num_bins::Int=38,
    distance_gaussian_dim::Int=32,
    miniformer_blocks::Bool=false,
    use_token_distance_feats::Bool=true,
)
    z_norm = BGLayerNorm(token_z; eps=1f-5)
    v_norm = BGLayerNorm(token_distance_dim; eps=1f-5)
    z_proj = LinearNoBias(token_z, token_distance_dim)
    a_proj = LinearNoBias(num_bins + (use_token_distance_feats ? 4 * token_z : 0), token_distance_dim)
    u_proj = LinearNoBias(token_distance_dim, token_z)
    Onion.torch_linear_init!(z_proj.weight)
    Onion.torch_linear_init!(a_proj.weight)
    Onion.torch_linear_init!(u_proj.weight)

    pairformer = if miniformer_blocks
        Onion.MiniformerNoSeqModule(token_distance_dim, token_distance_blocks; dropout=dropout, post_layer_norm=post_layer_norm, activation_checkpointing=activation_checkpointing)
    else
        Onion.PairformerNoSeqModule(token_distance_dim, token_distance_blocks; dropout=dropout, pairwise_head_width=pairwise_head_width, pairwise_num_heads=pairwise_num_heads, post_layer_norm=post_layer_norm, activation_checkpointing=activation_checkpointing)
    end

    token_distance_encoder = use_token_distance_feats ? DistanceTokenEncoder(distance_gaussian_dim, token_z, token_z) : nothing

    return TokenDistanceModule(min_dist, max_dist, num_bins, use_token_distance_feats, z_norm, v_norm, z_proj, a_proj, u_proj, pairformer, token_distance_encoder)
end

function (tm::TokenDistanceModule)(z, feats, pair_mask, relative_position_encoding; use_kernels::Bool=false)
    token_distance_mask = feats["token_distance_mask"]
    token_coords = feats["center_coords"]

    coords_b = permutedims(token_coords, (3,2,1))
    dists = pairwise_distance_batch(coords_b)
    boundaries_cpu = Float32.(collect(range(tm.min_dist, tm.max_dist; length=tm.num_bins-1)))
    boundaries_dev = copyto!(similar(dists, Float32, length(boundaries_cpu)), boundaries_cpu)
    distogram = sum(reshape(dists, size(dists,1), size(dists,2), size(dists,3), 1) .> reshape(boundaries_dev, 1, 1, 1, :); dims=4)
    distogram = dropdims(distogram; dims=4)
    distogram = one_hot(Int.(distogram), tm.num_bins)

    if tm.use_token_distance_feats
        dist_features = tm.token_distance_encoder(relative_position_encoding, feats)
        dist_features_b = permutedims(dist_features, (4,2,3,1))
        a_ij = cat(distogram, dist_features_b; dims=4)
    else
        a_ij = distogram
    end

    mask_b = permutedims(token_distance_mask, (3,1,2))
    a_ij = a_ij .* reshape(mask_b, size(mask_b,1), size(mask_b,2), size(mask_b,3), 1)

    a_ij = tm.a_proj(permutedims(a_ij, (4,2,3,1)))
    a_ij = permutedims(a_ij, (4,2,3,1))

    v = tm.z_proj(tm.z_norm(z))
    v = v .+ permutedims(a_ij, (4,2,3,1))
    v = v .+ tm.pairformer(v, pair_mask; use_kernels=use_kernels)
    v = tm.v_norm(v)

    u = tm.u_proj(NNlib.relu.(v))
    return u
end

@concrete struct MSALayer <: Onion.Layer
    msa_dropout
    msa_transition
    pair_weighted_averaging
    pairformer_layer
    outer_product_mean
end

@layer MSALayer

function MSALayer(
    msa_s::Int,
    token_z::Int,
    msa_dropout::Float32,
    z_dropout::Float32;
    miniformer_blocks::Bool=true,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
)
    msa_transition = Onion.Transition(msa_s, msa_s * 4)
    pair_weighted_averaging = Onion.PairWeightedAveraging(msa_s, token_z, 32, 8)
    pairformer_layer = if miniformer_blocks
        Onion.MiniformerNoSeqLayer(token_z; dropout=z_dropout)
    else
        Onion.PairformerNoSeqLayer(token_z; dropout=z_dropout, pairwise_head_width=pairwise_head_width, pairwise_num_heads=pairwise_num_heads)
    end
    outer_product_mean = Onion.OuterProductMean(msa_s, 32, token_z)

    return MSALayer(msa_dropout, msa_transition, pair_weighted_averaging, pairformer_layer, outer_product_mean)
end

function (layer::MSALayer)(z, m, token_mask, msa_mask;
    chunk_heads_pwa::Bool=false,
    chunk_size_transition_z=nothing,
    chunk_size_transition_msa=nothing,
    chunk_size_outer_product=nothing,
    chunk_size_tri_attn=nothing,
    use_kernels::Bool=false,
)
    msa_dropout = Onion.get_dropout_mask(layer.msa_dropout, m, Onion.bg_istraining())
    m = m .+ msa_dropout .* layer.pair_weighted_averaging(m, z, token_mask; chunk_heads=chunk_heads_pwa)
    m = m .+ layer.msa_transition(m; chunk_size=chunk_size_transition_msa)

    z = z .+ layer.outer_product_mean(m, msa_mask; chunk_size=chunk_size_outer_product)
    z = layer.pairformer_layer(z, token_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)

    return z, m
end

@concrete struct MSAModule <: Onion.Layer
    msa_blocks::Int
    msa_dropout
    z_dropout
    use_paired_feature::Bool
    activation_checkpointing::Bool
    s_proj
    msa_proj
    layers
end

@layer MSAModule

function MSAModule(
    msa_s::Int,
    token_z::Int,
    token_s::Int,
    msa_blocks::Int,
    msa_dropout::Float32,
    z_dropout::Float32;
    miniformer_blocks::Bool=true,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    activation_checkpointing::Bool=false,
    use_paired_feature::Bool=false,
)
    s_proj = LinearNoBias(token_s, msa_s)
    msa_proj = LinearNoBias(num_tokens + 2 + (use_paired_feature ? 1 : 0), msa_s)
    Onion.torch_linear_init!(s_proj.weight)
    Onion.torch_linear_init!(msa_proj.weight)

    layers = [
        MSALayer(msa_s, token_z, msa_dropout, z_dropout; miniformer_blocks=miniformer_blocks, pairwise_head_width=pairwise_head_width, pairwise_num_heads=pairwise_num_heads)
        for _ in 1:msa_blocks
    ]

    return MSAModule(msa_blocks, msa_dropout, z_dropout, use_paired_feature, activation_checkpointing, s_proj, msa_proj, layers)
end

function (m::MSAModule)(z, emb, feats; use_kernels::Bool=false)
    # set chunk sizes
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

    msa = feats["msa"]
    msa = one_hot(Int.(msa), num_tokens)
    msa = permutedims(msa, (4, 1, 2, 3))
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
        z, m_feat = layer(z, m_feat, token_mask, msa_mask; chunk_heads_pwa=chunk_heads_pwa, chunk_size_transition_z=chunk_size_transition_z, chunk_size_transition_msa=chunk_size_transition_msa, chunk_size_outer_product=chunk_size_outer_product, chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)
    end

    return z
end

@concrete struct BFactorModule <: Onion.Layer
    bfactor
end

@layer BFactorModule

function BFactorModule(token_s::Int, num_bins::Int)
    bfactor = BGLinear(token_s, num_bins; bias=true, init=:default)
    Onion.torch_linear_init!(bfactor.weight, bfactor.bias)
    return BFactorModule(bfactor)
end

function (b::BFactorModule)(s)
    return b.bfactor(s)
end

@concrete struct DistogramModule <: Onion.Layer
    distogram
    num_bins::Int
end

@layer DistogramModule

function DistogramModule(token_z::Int, num_bins::Int)
    distogram = BGLinear(token_z, num_bins; bias=true, init=:default)
    Onion.torch_linear_init!(distogram.weight, distogram.bias)
    return DistogramModule(distogram, num_bins)
end

function (d::DistogramModule)(z)
    z_sym = z .+ permutedims(z, (1,3,2,4))
    out = d.distogram(z_sym)
    return reshape(out, size(out,1), size(out,2), size(out,3), 1, size(out,4))
end
