using Onion
using NNlib
using Statistics: mean, var
using Printf

const BGLayerNorm = Onion.BGLayerNorm
const BGLinear = Onion.BGLinear

exists(x) = x !== nothing

function default(x, d)
    return exists(x) ? x : d
end

bg_log(x; eps=1f-20) = log.(max.(x, eps))

silu(x) = x .* NNlib.sigmoid.(x)

struct SwiGLU <: Onion.Layer end
# SwiGLU is stateless (no fields) — override adapt_structure to prevent
# Flux.gpu() infinite recursion (the @layer Layer in Onion triggers fmap on all subtypes)
Onion.Flux.Adapt.adapt_structure(to, x::SwiGLU) = x

function (s::SwiGLU)(x)
    c = size(x, 1)
    @assert c % 2 == 0 "SwiGLU expects even feature size"
    split = c ÷ 2
    x2d = reshape(x, c, :)
    x1 = view(x2d, 1:split, :)
    x2 = view(x2d, split+1:c, :)
    y = silu(x2) .* x1
    return reshape(y, split, size(x)[2:end]...)
end

# Linear without bias, feature-first
LinearNoBias(in_dim::Int, out_dim::Int; init::Symbol=:default) = BGLinear(in_dim, out_dim; bias=false, init=init)

# Non-affine LayerNorm, feature-first
@concrete struct BGNorm <: Onion.Layer
    eps
end

@layer BGNorm

function BGNorm(; eps=1f-5)
    return BGNorm(eps)
end

function (ln::BGNorm)(x)
    μ = mean(x; dims=1)
    σ² = var(x; dims=1, mean=μ, corrected=false)
    return (x .- μ) ./ sqrt.(σ² .+ ln.eps)
end

# LayerNorm with weight but no bias, feature-first
@concrete struct BGWeightNorm <: Onion.Layer
    weight
    eps
end

@layer BGWeightNorm

function BGWeightNorm(c_in::Int; eps=1f-5)
    weight = ones(Float32, c_in)
    return BGWeightNorm(weight, eps)
end

function (ln::BGWeightNorm)(x)
    μ = mean(x; dims=1)
    σ² = var(x; dims=1, mean=μ, corrected=false)
    y = (x .- μ) ./ sqrt.(σ² .+ ln.eps)
    w = reshape(ln.weight, length(ln.weight), ntuple(_ -> 1, ndims(x)-1)...)
    return y .* w
end

# Gaussian smearing
@concrete struct GaussianSmearing <: Onion.Layer
    offset
    coeff
end

@layer GaussianSmearing

function GaussianSmearing(; start=0.0f0, stop=5.0f0, num_gaussians::Int=50)
    offset = range(Float32(start), Float32(stop); length=num_gaussians)
    coeff = -0.5f0 / (offset[2] - offset[1])^2
    return GaussianSmearing(collect(offset), coeff)
end

function (g::GaussianSmearing)(dist)
    # dist can be any shape; broadcast over a new trailing gaussian dimension
    off = reshape(g.offset, ntuple(_ -> 1, ndims(dist) - 1)..., length(g.offset))
    d = dist .- off
    return exp.(g.coeff .* (d .^ 2))
end

@concrete struct GaussianRandom3DEncodings <: Onion.Layer
    center
    std
    dim::Int
end

@layer GaussianRandom3DEncodings

function GaussianRandom3DEncodings(dim::Int=50)
    center = randn(Float32, dim, 3)
    std = rand(Float32, dim)
    return GaussianRandom3DEncodings(center, std, dim)
end

function (g::GaussianRandom3DEncodings)(coords)
    # coords: (3, N, B)
    B = size(coords, 3)
    N = size(coords, 2)
    # (dim, 3) -> (dim, 3, 1, 1)
    center = reshape(g.center, g.dim, 3, 1, 1)
    std = reshape(g.std, g.dim, 1, 1)
    coords2 = reshape(coords, 1, 3, N, B)
    dist2 = sum((coords2 .- center).^2; dims=2)
    dist2 = reshape(dist2, g.dim, N, B)
    return exp.(-dist2 ./ (2 .* std))
end

# Rotation utilities
function _copysign(a, b)
    return ifelse.((a .< 0) .!= (b .< 0), -a, a)
end

function quaternion_to_matrix(quat)
    # quat: (4, B)
    r = quat[1, :]
    i = quat[2, :]
    j = quat[3, :]
    k = quat[4, :]
    two_s = 2f0 ./ sum(quat .* quat; dims=1)
    o11 = 1 .- two_s .* (j .* j .+ k .* k)
    o12 = two_s .* (i .* j .- k .* r)
    o13 = two_s .* (i .* k .+ j .* r)
    o21 = two_s .* (i .* j .+ k .* r)
    o22 = 1 .- two_s .* (i .* i .+ k .* k)
    o23 = two_s .* (j .* k .- i .* r)
    o31 = two_s .* (i .* k .- j .* r)
    o32 = two_s .* (j .* k .+ i .* r)
    o33 = 1 .- two_s .* (i .* i .+ j .* j)
    R = Array{Float32}(undef, 3, 3, size(quat, 2))
    R[1,1,:] = o11
    R[1,2,:] = o12
    R[1,3,:] = o13
    R[2,1,:] = o21
    R[2,2,:] = o22
    R[2,3,:] = o23
    R[3,1,:] = o31
    R[3,2,:] = o32
    R[3,3,:] = o33
    return R
end

function random_quaternions(n::Int)
    o = randn(Float32, 4, n)
    s = sum(o .* o; dims=1)
    o = o ./ _copysign(sqrt.(s), o[1:1, :])
    return o
end

function random_rotations(n::Int)
    q = random_quaternions(n)
    return quaternion_to_matrix(q)
end

function rotate_coords(coords, R)
    # coords: (3, M, B), R: (3, 3, B)
    # Apply R^T @ coords per batch, using batched_mul for GPU compatibility
    return NNlib.batched_mul(permutedims(R, (2, 1, 3)), coords)
end

function center(atom_coords, atom_mask)
    # atom_coords: (3, M, B), atom_mask: (M, B)
    mask = reshape(atom_mask, 1, size(atom_mask, 1), size(atom_mask, 2))
    denom = sum(mask; dims=2)
    mean_coords = sum(atom_coords .* mask; dims=2) ./ denom
    return atom_coords .- mean_coords
end

function compute_random_augmentation(multiplicity; s_trans=1.0f0)
    R = random_rotations(multiplicity)
    random_trans = randn(Float32, 3, 1, multiplicity) .* Float32(s_trans)
    return R, random_trans
end

function compute_random_augmentation(multiplicity, device_ref::AbstractArray; s_trans=1.0f0)
    R_cpu, t_cpu = compute_random_augmentation(multiplicity; s_trans=s_trans)
    R = copyto!(similar(device_ref, Float32, size(R_cpu)), R_cpu)
    t = copyto!(similar(device_ref, Float32, size(t_cpu)), t_cpu)
    return R, t
end

function repeat_interleave_batch(x, m::Int)
    m == 1 && return x
    bdim = ndims(x)
    B = size(x, bdim)
    idx = repeat(collect(1:B), inner=m)
    return x[ntuple(_ -> Colon(), bdim-1)..., idx]
end

function repeat_interleave_dim(x, m::Int, dim::Int)
    m == 1 && return x
    idx = repeat(collect(1:size(x, dim)), inner=m)
    inds = ntuple(i -> i == dim ? idx : Colon(), ndims(x))
    return x[inds...]
end

function torch_embedding_init!(weight)
    bound = 1f0 / sqrt(size(weight, 2))
    weight .= rand(Float32, size(weight)) .* (2f0 * bound) .- bound
    return weight
end

function center_random_augmentation(
    atom_coords,
    atom_mask;
    s_trans=1.0f0,
    augmentation::Bool=true,
    centering::Bool=true,
    return_second_coords::Bool=false,
    second_coords=nothing,
)
    coords = atom_coords
    other = second_coords
    if centering
        mask = reshape(atom_mask, 1, size(atom_mask, 1), size(atom_mask, 2))
        denom = sum(mask; dims=2)
        mean_coords = sum(coords .* mask; dims=2) ./ denom
        coords = coords .- mean_coords
        if other !== nothing
            other = other .- mean_coords
        end
    end

    if augmentation
        R = random_rotations(size(coords, 3))
        coords = rotate_coords(coords, R)
        if other !== nothing
            other = rotate_coords(other, R)
        end
        random_trans = randn(Float32, 3, 1, size(coords, 3)) .* Float32(s_trans)
        coords = coords .+ random_trans
        if other !== nothing
            other = other .+ random_trans
        end
    end

    if return_second_coords
        return coords, other
    end
    return coords
end

# Inverse CDF (PPF) for standard normal using Acklam's approximation.
function norm_ppf(p::Real)
    if p <= 0
        return -Inf
    elseif p >= 1
        return Inf
    end
    # Coefficients in rational approximations.
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    plow = 0.02425
    phigh = 1 - plow

    if p < plow
        q = sqrt(-2 * log(p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        return num / den
    elseif p <= phigh
        q = p - 0.5
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
        return num / den
    else
        q = sqrt(-2 * log(1 - p))
        num = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        return num / den
    end
end

# Regularized incomplete beta CDF using continued fraction (Numerical Recipes).
function _betacf(a::Float64, b::Float64, x::Float64)
    MAXIT = 200
    EPS = 3.0e-7
    FPMIN = 1.0e-30

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN
        d = FPMIN
    end
    d = 1.0 / d
    h = d

    for m in 1:MAXIT
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN
            d = FPMIN
        end
        c = 1.0 + aa / c
        if abs(c) < FPMIN
            c = FPMIN
        end
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN
            d = FPMIN
        end
        c = 1.0 + aa / c
        if abs(c) < FPMIN
            c = FPMIN
        end
        d = 1.0 / d
        del = d * c
        h *= del

        if abs(del - 1.0) < EPS
            break
        end
    end
    return h
end

function beta_cdf(x::Real, a::Real, b::Real)
    if x <= 0
        return 0.0
    elseif x >= 1
        return 1.0
    end
    a64 = Float64(a)
    b64 = Float64(b)
    x64 = Float64(x)
    # Compute regularized incomplete beta.
    bt = exp(_loggamma(a64 + b64) - _loggamma(a64) - _loggamma(b64) + a64 * log(x64) + b64 * log1p(-x64))
    if x64 < (a64 + 1.0) / (a64 + b64 + 2.0)
        return bt * _betacf(a64, b64, x64) / a64
    else
        return 1.0 - bt * _betacf(b64, a64, 1.0 - x64) / b64
    end
end

# Lanczos approximation for log-gamma.
function _loggamma(z::Float64)
    # Reflection for z < 0.5
    if z < 0.5
        return log(pi) - log(sin(pi * z)) - _loggamma(1.0 - z)
    end
    # Lanczos coefficients, g=7, n=9
    p = (
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )
    z -= 1.0
    x = p[1]
    for i in 2:length(p)
        x += p[i] / (z + (i - 1))
    end
    g = 7.0
    t = z + g + 0.5
    return 0.5 * log(2.0 * pi) + (z + 0.5) * log(t) - t + log(x)
end

# Decode atom name chars (one-hot) into a string.
function decode_atom_name_chars(chars)
    if ndims(chars) != 2
        error("decode_atom_name_chars expects a 2D array, got dims=$(ndims(chars))")
    end
    if size(chars, 1) == 64 && size(chars, 2) == 4
        idxs = [argmax(view(chars, :, i)) - 1 for i in 1:4]
    elseif size(chars, 1) == 4 && size(chars, 2) == 64
        idxs = [argmax(view(chars, i, :)) - 1 for i in 1:4]
    else
        error("decode_atom_name_chars expects shape (64,4) or (4,64), got $(size(chars))")
    end
    codes = Int[]
    for idx in idxs
        if idx != 0
            push!(codes, idx + 32)
        end
    end
    return String(Char.(codes))
end

# Encode atom name string into one-hot char tensor (64, 4).
function encode_atom_name_chars(name::AbstractString)
    name = uppercase(strip(String(name)))
    codes = [Int(c) - 32 for c in collect(name)]
    if length(codes) < 4
        append!(codes, zeros(Int, 4 - length(codes)))
    elseif length(codes) > 4
        codes = codes[1:4]
    end
    out = zeros(Float32, 64, 4)
    @inbounds for i in 1:4
        idx = codes[i]
        if 0 <= idx < 64
            out[idx + 1, i] = 1f0
        end
    end
    return out
end

# Build masked ref atom name chars like python's convert_atom_name(mask_element + i)
function build_masked_ref_atom_name_chars(atom_to_token, atom_pad_mask; mask_element::String="FL")
    # atom_to_token: (M, T, B), atom_pad_mask: (M, B)
    M = size(atom_to_token, 1)
    T = size(atom_to_token, 2)
    B = size(atom_to_token, 3)
    out = zeros(Float32, 64, 4, M, B)
    for b in 1:B
        counts = zeros(Int, T)
        for m in 1:M
            atom_pad_mask[m, b] > 0.5 || continue
            t = argmax(view(atom_to_token, m, :, b))
            i = counts[t]
            counts[t] += 1
            name = string(mask_element, i)
            out[:, :, m, b] .= encode_atom_name_chars(name)
        end
    end
    return out
end

# Compute backbone mask from atom name chars (N, CA, C, O)
function compute_backbone_mask(ref_atom_name_chars)
    # ref_atom_name_chars: (64, 4, M, B)
    M = size(ref_atom_name_chars, 3)
    B = size(ref_atom_name_chars, 4)
    mask = zeros(Float32, M, B)
    for b in 1:B
        for m in 1:M
            name = decode_atom_name_chars(view(ref_atom_name_chars, :, :, m, b))
            if name == "N" || name == "CA" || name == "C" || name == "O"
                mask[m, b] = 1f0
            end
        end
    end
    return mask
end

function chain_id_from_asym(asym_id::Integer)
    alphabet = [collect('A':'Z'); collect('a':'z'); collect('0':'9')]
    idx = Int(asym_id) + 1
    if idx < 1
        idx = 1
    end
    return alphabet[mod1(idx, length(alphabet))]
end

# Weighted rigid alignment (feature-first coords: (3, M, B)).
function weighted_rigid_align(true_coords, pred_coords, weights, mask)
    # Feature-first coords: (3, M, B)
    w = reshape(weights .* mask, 1, size(weights, 1), size(weights, 2)) # (1, M, B)
    wsum = sum(w; dims=2)
    true_centroid = sum(true_coords .* w; dims=2) ./ wsum
    pred_centroid = sum(pred_coords .* w; dims=2) ./ wsum

    tc_centered = true_coords .- true_centroid
    pc_centered = pred_coords .- pred_centroid

    pcw = pc_centered .* w
    cov = NNlib.batched_mul(pcw, permutedims(tc_centered, (2, 1, 3))) # (3,3,B)

    # SVD requires CPU; compute rotation matrices on CPU then transfer back
    cov_cpu = Array(cov)
    B = size(cov_cpu, 3)
    rot_cpu = Array{Float32}(undef, 3, 3, B)
    for b in 1:B
        cov_b = Float32.(cov_cpu[:, :, b])
        U, _, V = svd(cov_b)
        Vt = V'
        rot_b = U * Vt
        F = Matrix{Float32}(I, 3, 3)
        F[3, 3] = det(rot_b)
        rot_cpu[:, :, b] = U * F * Vt
    end

    # Transfer rotation to same device as input and apply via batched_mul
    rot = copyto!(similar(true_coords, Float32, size(rot_cpu)), rot_cpu)
    aligned = NNlib.batched_mul(rot, tc_centered) .+ pred_centroid
    return aligned
end

function weighted_rigid_centering(true_coords, pred_coords, weights, mask)
    w = reshape(weights .* mask, 1, size(weights, 1), size(weights, 2))
    wsum = sum(w; dims=2)
    true_centroid = sum(true_coords .* w; dims=2) ./ wsum
    pred_centroid = sum(pred_coords .* w; dims=2) ./ wsum
    tc_centered = true_coords .- true_centroid
    return tc_centered .+ pred_centroid
end
