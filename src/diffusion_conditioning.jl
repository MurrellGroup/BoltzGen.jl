using Onion

const BGLayerNorm = Onion.BGLayerNorm

@concrete struct DiffusionConditioning <: Onion.Layer
    pairwise_conditioner
    atom_encoder
    atom_enc_proj_z
    atom_dec_proj_z
    token_trans_proj_z
end

@layer DiffusionConditioning

function DiffusionConditioning(
    token_s::Int,
    token_z::Int,
    atom_s::Int,
    atom_z::Int;
    atoms_per_window_queries::Int=32,
    atoms_per_window_keys::Int=128,
    atom_encoder_depth::Int=3,
    atom_encoder_heads::Int=4,
    token_transformer_depth::Int=24,
    token_transformer_heads::Int=8,
    atom_decoder_depth::Int=3,
    atom_decoder_heads::Int=4,
    atom_feature_dim::Int=128,
    conditioning_transition_layers::Int=2,
)
    pairwise_conditioner = PairwiseConditioning(token_z, token_z; num_transitions=conditioning_transition_layers)

    atom_encoder = AtomEncoder(
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim;
        structure_prediction=true,
    )

    atom_enc_proj_z = [
        (BGLayerNorm(atom_z; eps=1f-5), LinearNoBias(atom_z, atom_encoder_heads))
        for _ in 1:atom_encoder_depth
    ]
    for layer in atom_enc_proj_z
        Onion.torch_linear_init!(layer[2].weight)
    end

    atom_dec_proj_z = [
        (BGLayerNorm(atom_z; eps=1f-5), LinearNoBias(atom_z, atom_decoder_heads))
        for _ in 1:atom_decoder_depth
    ]
    for layer in atom_dec_proj_z
        Onion.torch_linear_init!(layer[2].weight)
    end

    token_trans_proj_z = [
        (BGLayerNorm(token_z; eps=1f-5), LinearNoBias(token_z, token_transformer_heads))
        for _ in 1:token_transformer_depth
    ]
    for layer in token_trans_proj_z
        Onion.torch_linear_init!(layer[2].weight)
    end

    return DiffusionConditioning(pairwise_conditioner, atom_encoder, atom_enc_proj_z, atom_dec_proj_z, token_trans_proj_z)
end

function (dc::DiffusionConditioning)(; s_trunk, z_trunk, relative_position_encoding, feats)
    z = dc.pairwise_conditioner(z_trunk, relative_position_encoding)
    q, c, p, to_keys = dc.atom_encoder(feats; s_trunk=s_trunk, z=z)

    atom_enc_bias = map(layer -> layer[2](layer[1](p)), dc.atom_enc_proj_z)
    atom_enc_bias = cat(atom_enc_bias...; dims=1)

    atom_dec_bias = map(layer -> layer[2](layer[1](p)), dc.atom_dec_proj_z)
    atom_dec_bias = cat(atom_dec_bias...; dims=1)

    token_trans_bias = map(layer -> layer[2](layer[1](z)), dc.token_trans_proj_z)
    token_trans_bias = cat(token_trans_bias...; dims=1)

    return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias
end
