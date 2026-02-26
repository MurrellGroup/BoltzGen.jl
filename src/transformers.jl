using Onion
using NNlib

const BGLayerNorm = Onion.BGLayerNorm

@concrete struct AdaLN <: Onion.Layer
    a_norm
    s_norm
    s_scale
    s_bias
end

@layer AdaLN

function AdaLN(dim::Int, dim_single_cond::Int)
    a_norm = BGNorm()
    s_norm = BGWeightNorm(dim_single_cond)
    s_scale = BGLinear(dim_single_cond, dim; bias=true, init=:default)
    s_bias = LinearNoBias(dim_single_cond, dim)
    Onion.torch_linear_init!(s_scale.weight, s_scale.bias)
    Onion.torch_linear_init!(s_bias.weight)
    return AdaLN(a_norm, s_norm, s_scale, s_bias)
end

function (ln::AdaLN)(a, s)
    a_hat = ln.a_norm(a)
    s_hat = ln.s_norm(s)
    scale = NNlib.sigmoid.(ln.s_scale(s_hat))
    bias = ln.s_bias(s_hat)
    return scale .* a_hat .+ bias
end

@concrete struct ConditionedTransitionBlock <: Onion.Layer
    adaln
    swish_gate
    a_to_b
    b_to_a
    output_projection
end

@layer ConditionedTransitionBlock

function ConditionedTransitionBlock(dim_single::Int, dim_single_cond::Int; expansion_factor::Int=2)
    adaln = AdaLN(dim_single, dim_single_cond)
    dim_inner = dim_single * expansion_factor
    swish_gate = (LinearNoBias(dim_single, dim_inner * 2), SwiGLU())
    a_to_b = LinearNoBias(dim_single, dim_inner)
    b_to_a = LinearNoBias(dim_inner, dim_single)

    Onion.torch_linear_init!(swish_gate[1].weight)
    Onion.torch_linear_init!(a_to_b.weight)
    Onion.torch_linear_init!(b_to_a.weight)

    output_linear = BGLinear(dim_single_cond, dim_single; bias=true, init=:default)
    # match torch init: weight zeros, bias -2
    output_linear.weight .= 0f0
    output_linear.bias .= -2f0
    output_projection = output_linear

    return ConditionedTransitionBlock(adaln, swish_gate, a_to_b, b_to_a, output_projection)
end

function (blk::ConditionedTransitionBlock)(a, s)
    a2 = blk.adaln(a, s)
    gate_lin, swiglu = blk.swish_gate
    b = swiglu(gate_lin(a2)) .* blk.a_to_b(a2)
    out_scale = NNlib.sigmoid.(blk.output_projection(s))
    a_out = out_scale .* blk.b_to_a(b)
    return a_out
end

@concrete struct DiffusionTransformerLayer <: Onion.Layer
    adaln
    pair_bias_attn
    output_projection
    transition
    post_lnorm
end

@layer DiffusionTransformerLayer

function DiffusionTransformerLayer(
    heads::Int;
    dim::Int=384,
    dim_single_cond::Int=dim,
    post_layer_norm::Bool=false,
    use_qk_norm::Bool=false,
)
    adaln = AdaLN(dim, dim_single_cond)
    pair_bias_attn = Onion.AttentionPairBias(dim, dim, heads; compute_pair_bias=false, use_qk_norm=use_qk_norm)

    output_linear = BGLinear(dim_single_cond, dim; bias=true, init=:default)
    output_linear.weight .= 0f0
    output_linear.bias .= -2f0
    output_projection = output_linear

    transition = ConditionedTransitionBlock(dim, dim_single_cond)
    post_lnorm = post_layer_norm ? BGLayerNorm(dim; eps=1f-5) : identity

    return DiffusionTransformerLayer(adaln, pair_bias_attn, output_projection, transition, post_lnorm)
end

function (layer::DiffusionTransformerLayer)(a, s, bias, mask, to_keys, multiplicity::Int)
    b = layer.adaln(a, s)
    # Use if-expression to avoid variable reassignment that confuses Zygote's
    # gradient accumulation (k_in and b have different shapes when to_keys != nothing)
    k_in, mask_k = if to_keys !== nothing
        (to_keys(b), to_keys(mask))
    else
        (b, mask)
    end

    b2 = layer.pair_bias_attn(b, bias, mask_k, k_in; multiplicity=multiplicity)
    scale = NNlib.sigmoid.(layer.output_projection(s))
    b2_scaled = scale .* b2

    a2 = a .+ b2_scaled
    a3 = a2 .+ layer.transition(a2, s)
    return layer.post_lnorm(a3)
end

@concrete struct DiffusionTransformer <: Onion.Layer
    layers
    activation_checkpointing::Bool
end

@layer DiffusionTransformer

function DiffusionTransformer(; depth::Int, heads::Int, dim::Int=384, dim_single_cond::Int=dim, use_qk_norm::Bool=false, activation_checkpointing::Bool=false, post_layer_norm::Bool=false)
    layers = [
        DiffusionTransformerLayer(heads; dim=dim, dim_single_cond=dim_single_cond, post_layer_norm=post_layer_norm, use_qk_norm=use_qk_norm)
        for _ in 1:depth
    ]
    return DiffusionTransformer(layers, activation_checkpointing)
end

function (t::DiffusionTransformer)(a, s; bias, mask, to_keys=nothing, multiplicity::Int=1, use_uniform_bias::Bool=false)
    L = length(t.layers)

    if use_uniform_bias
        for (i, layer) in enumerate(t.layers)
            a = layer(a, s, bias, mask, to_keys, multiplicity)
        end
    else
        D = size(bias, 1)
        @assert D % L == 0 "bias feature dim must be divisible by layers"
        bias_reshaped = reshape(bias, D ÷ L, L, size(bias, 2), size(bias, 3), size(bias, 4))
        for (i, layer) in enumerate(t.layers)
            bias_l = view(bias_reshaped, :, i, :, :, :)
            a = layer(a, s, bias_l, mask, to_keys, multiplicity)
        end
    end
    return a
end

@concrete struct AtomTransformer <: Onion.Layer
    attn_window_queries::Int
    attn_window_keys::Int
    diffusion_transformer
end

@layer AtomTransformer

function AtomTransformer(; attn_window_queries::Int, attn_window_keys::Int, diffusion_transformer_kwargs...)
    diffusion_transformer = DiffusionTransformer(; diffusion_transformer_kwargs...)
    return AtomTransformer(attn_window_queries, attn_window_keys, diffusion_transformer)
end

function (t::AtomTransformer)(q, c; bias, to_keys, mask, multiplicity::Int=1)
    W = t.attn_window_queries
    H = t.attn_window_keys

    C = size(q, 1)
    N = size(q, 2)
    Bq = size(q, 3)
    NW = N ÷ W

    # Reshape into windows — use unique variable names to avoid Zygote variable-reassignment
    # bugs where gradient accumulation across different shapes causes DimensionMismatch.
    # (C, N, B) -> (C, W, NW, B) -> (C, W, B, NW) -> (C, W, B*NW)
    qw4 = reshape(q, C, W, NW, Bq)
    cw4 = reshape(c, size(c,1), W, NW, Bq)
    mw3 = reshape(mask, W, NW, Bq)

    qw4p = permutedims(qw4, (1, 2, 4, 3))
    cw4p = permutedims(cw4, (1, 2, 4, 3))
    mw3p = permutedims(mw3, (1, 3, 2))

    qw = reshape(qw4p, C, W, Bq * NW)
    cw = reshape(cw4p, size(c,1), W, Bq * NW)
    mw = reshape(mw3p, W, Bq * NW)

    # bias is (D, W, H, K, B) — unique names to avoid shape-changing reassignment
    bias_eff = multiplicity != 1 ? repeat_interleave_batch(bias, multiplicity) : bias
    bias_perm = permutedims(bias_eff, (1, 2, 3, 5, 4))
    Bbias = size(bias_eff, 5)
    bias_w = reshape(bias_perm, size(bias_eff,1), W, H, Bbias * NW)

    function to_keys_new(x)
        if ndims(x) == 2
            x3 = reshape(x, 1, W * NW, Bq)
            y = to_keys(x3)
            return reshape(y, size(y,2), size(y,3) * size(y,4))
        else
            x3 = reshape(x, size(x,1), W * NW, Bq)
            y = to_keys(x3)
            return reshape(y, size(y,1), size(y,2), size(y,3) * size(y,4))
        end
    end

    out_flat = t.diffusion_transformer(qw, cw; bias=bias_w, mask=mw, to_keys=to_keys_new, multiplicity=1)

    # reshape back: (C, W, B*NW) -> (C, W, B, NW) -> (C, W, NW, B) -> (C, N, B)
    out4 = reshape(out_flat, C, W, Bq, NW)
    out4p = permutedims(out4, (1, 2, 4, 3))
    return reshape(out4p, C, N, Bq)
end
