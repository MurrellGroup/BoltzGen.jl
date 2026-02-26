# ── High-level design API for differentiable Boltz2 ────────────────────────────
#
# Provides setup_boltz2_design and make_boltz2_loss_function for gradient-based
# binder hallucination using the differentiable Boltz2 forward path.

"""
    setup_boltz2_design(handle, target_seq, binder_length; target_msa_path=nothing, kwargs...)

Build features for binder design and prepare model + features on GPU.

Arguments:
- `handle`: BoltzGenHandle (from `load_boltz2()`)
- `target_seq`: target protein sequence (String)
- `binder_length`: number of residues to design
- `target_msa_path`: optional path to target MSA file (a3m format)

Returns a NamedTuple `(model, feats, binder_length)` ready for
`boltz2_differentiable_forward` or `make_boltz2_loss_function`.
"""
function setup_boltz2_design(
    handle,
    target_seq::AbstractString,
    binder_length::Int;
    target_msa_path=nothing,
    max_msa_rows::Union{Nothing,Int}=nothing,
    design_mask_binder::Bool=true,
    mask_features::Bool=true,
    kwargs...
)
    # Build residue tokens: binder ("X" * binder_length) + target
    binder_tokens = fill("UNK", binder_length)
    target_tokens = [string(prot_letter_to_token[c]) for c in target_seq]
    residue_tokens = vcat(binder_tokens, target_tokens)
    T = length(residue_tokens)
    target_len = length(target_tokens)

    # Setup chain IDs: binder = chain 0, target = chain 1
    asym_ids = vcat(fill(0, binder_length), fill(1, target_len))
    entity_ids = vcat(fill(0, binder_length), fill(1, target_len))
    sym_ids = vcat(fill(0, binder_length), fill(0, target_len))
    mol_types = fill(chain_type_ids["PROTEIN"], T)

    # Design mask: binder positions are designable
    design_mask = vcat(fill(true, binder_length), fill(false, target_len))
    chain_design_mask = design_mask  # same for single-chain binder

    # Per-chain MSA concatenation (mirrors yaml_parser.jl:1976-2049)
    # Binder has no MSA; target has MSA from file. Concatenate with gaps at binder positions.
    msa_kwargs = Dict{Symbol,Any}()
    if target_msa_path !== nothing
        raw_rows = load_msa_sequences(target_msa_path)
        if max_msa_rows !== nothing
            raw_rows = raw_rows[1:min(end, max_msa_rows)]
        end
        if !isempty(raw_rows)
            S = length(raw_rows)
            binder_idxs = 1:binder_length
            target_idxs = (binder_length+1):T

            msa_rows = Vector{String}(undef, S)
            paired_rows = falses(S)
            paired_rows[1] = true
            has_del_rows = zeros(Float32, S, T)
            del_val_rows = zeros(Float32, S, T)

            for s in 1:S
                row_chars = fill('-', T)
                # Binder: first row gets query-like chars, rest get gaps
                if s == 1
                    for tidx in binder_idxs
                        row_chars[tidx] = 'X'  # UNK for designed binder
                    end
                end
                # Target: normalize and map to concatenated positions
                row_norm, has_del_chain, del_val_chain = _normalize_msa_row(raw_rows[s], target_len)
                for (k, tidx) in enumerate(target_idxs)
                    row_chars[tidx] = row_norm[k]
                    has_del_rows[s, tidx] = has_del_chain[k]
                    del_val_rows[s, tidx] = del_val_chain[k]
                end
                msa_rows[s] = String(row_chars)
            end

            msa_kwargs[:msa_sequences] = msa_rows
            msa_kwargs[:msa_paired_rows] = paired_rows
            msa_kwargs[:msa_has_deletion_rows] = has_del_rows
            msa_kwargs[:msa_deletion_value_rows] = del_val_rows
            msa_kwargs[:target_msa_mask] = vcat(fill(false, binder_length), fill(true, target_len))
        end
    end

    # Build features using existing API
    feats = build_design_features(
        residue_tokens;
        mol_types=mol_types,
        asym_ids=asym_ids,
        entity_ids=entity_ids,
        sym_ids=sym_ids,
        design_mask=design_mask,
        chain_design_mask=chain_design_mask,
        max_msa_rows=max_msa_rows,
        msa_kwargs...,
        kwargs...
    )

    # Apply masking (sets binder res_type/profile to UNK, etc.)
    if mask_features
        feats = boltz_masker(feats; mask=true)
    end

    # Pre-one-hot the MSA for the differentiable path
    # feats["msa"]: (S, T, B) with integer token indices (0-indexed)
    msa_int = feats["msa"]
    # one_hot expects Integer input; these are the fixed (non-differentiable) MSA rows
    msa_onehot = one_hot(Int.(msa_int), num_tokens)  # (S, T, B, num_tokens)
    msa_onehot = permutedims(msa_onehot, (4, 1, 2, 3))  # (num_tokens, S, T, B)
    feats["msa_onehot"] = msa_onehot

    # Transfer to GPU
    feats_gpu = _dict_to_gpu(feats)
    model_gpu = Onion.Flux.gpu(handle.model)

    return (model=model_gpu, feats=feats_gpu, binder_length=binder_length)
end

"""
    init_soft_sequence(binder_length; init_mode=:gumbel, temperature=0.5, device=nothing)

Initialize a soft sequence for design optimization.

Modes:
- `:uniform`: uniform distribution over 20 AAs
- `:gumbel`: `softmax(temperature * gumbel_noise)` (mosaic-style initialization)

Returns: (20, binder_length, 1) Float32 array.
"""
function init_soft_sequence(binder_length::Int; init_mode::Symbol=:gumbel, temperature::Float32=0.5f0, device=nothing)
    if init_mode == :uniform
        soft_seq = fill(1f0/20f0, 20, binder_length, 1)
    elseif init_mode == :gumbel
        # Mosaic: softmax(temperature * gumbel_noise)
        gumbel = -log.(-log.(rand(Float32, 20, binder_length, 1) .+ 1f-20) .+ 1f-20)
        soft_seq = NNlib.softmax(temperature .* gumbel; dims=1)
    else
        error("Unknown init_mode: $init_mode. Use :uniform or :gumbel")
    end

    if device !== nothing
        soft_seq = copyto!(similar(device, Float32, size(soft_seq)), soft_seq)
    end
    return soft_seq
end

"""
    make_boltz2_loss_function(setup; recycling_steps=0, num_sampling_steps=nothing, multiplicity=1, loss_fn)

Create a closure `(soft_seq; key=nothing) → scalar_loss` for gradient-based optimization.

Arguments:
- `setup`: NamedTuple from `setup_boltz2_design`
- `loss_fn`: `(out_dict, feats, binder_length, soft_seq) → scalar` defining the design objective.
  The 4th argument `soft_seq` is the current (20, binder_length, 1) soft sequence,
  needed for losses like MPNN that compute inner products with the sequence.
- `recycling_steps`, `num_sampling_steps`, `multiplicity`: forward pass parameters

Example loss functions:
```julia
# Maximize pLDDT (ignoring soft_seq)
loss_plddt(out, feats, bl, sq) = -mean(out["complex_plddt"])

# Composite with MPNN
loss_composite(out, feats, bl, sq) = -mean(out["complex_plddt"]) + mpnn_term(out, feats, bl, sq)
```
"""
function make_boltz2_loss_function(
    setup;
    recycling_steps::Int=0,
    num_sampling_steps=nothing,
    multiplicity::Int=1,
    step_scale=nothing,
    noise_scale=nothing,
    sampling_schedule=nothing,
    time_dilation=nothing,
    use_kernels::Bool=false,
    loss_fn,
)
    return function(soft_seq; key=nothing)
        feats = set_binder_sequence(soft_seq, setup.feats, setup.binder_length)
        out = boltz2_differentiable_forward(
            setup.model,
            feats;
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            multiplicity=multiplicity,
            key=key,
            step_scale=step_scale,
            noise_scale=noise_scale,
            sampling_schedule=sampling_schedule,
            time_dilation=time_dilation,
            use_kernels=use_kernels,
        )
        return loss_fn(out, setup.feats, setup.binder_length, soft_seq)
    end
end

"""
    decode_soft_sequence(soft_seq)

Convert a soft sequence (20, L, B) to a hard amino acid string using argmax.
Returns a Vector{String} of length B.
"""
function decode_soft_sequence(soft_seq)
    L = size(soft_seq, 2)
    B = size(soft_seq, 3)
    aa_letters = collect("ARNDCQEGHILKMFPSTWYV")  # AlphaFold canonical order (matches Boltz2 tokens)
    sequences = String[]
    soft_cpu = Array(soft_seq)
    for b in 1:B
        chars = Char[]
        for i in 1:L
            idx = argmax(view(soft_cpu, :, i, b))
            push!(chars, aa_letters[idx])
        end
        push!(sequences, String(chars))
    end
    return sequences
end
