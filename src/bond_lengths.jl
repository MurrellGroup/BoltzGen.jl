# Bond length validation for protein structures.
#
# Bond connectivity, expected lengths, sp2 overrides, and BondViolation type
# all come from ProtInterop. This file provides aliases for backward compatibility
# and the BoltzGen-specific checker functions.

# ── Aliases from ProtInterop ──────────────────────────────────────────────────

const _BACKBONE_BONDS = ProtInterop.BACKBONE_BONDS
const _SIDECHAIN_BONDS = ProtInterop.SIDECHAIN_BONDS
const _EXPECTED_BOND_LENGTHS = ProtInterop.EXPECTED_BOND_LENGTHS
const _SP2_BOND_LENGTH_OVERRIDES = ProtInterop.SP2_BOND_LENGTH_OVERRIDES
const _PEPTIDE_BOND_LENGTH = ProtInterop.PEPTIDE_BOND_LENGTH
# BondViolation is imported via `using ProtInterop: BondViolation` in BoltzGen.jl

# ── Main checker ────────────────────────────────────────────────────────────────

"""
    check_bond_lengths(result; batch=1, tol_low=0.9, tol_high=1.1)

Check all protein backbone and sidechain covalent bond lengths against expected
values derived from reference geometry. Returns a named tuple with:

- `n_bonds_checked`: total bonds examined
- `n_violations`: number of bonds outside tolerance
- `n_residues_with_violations`: residues with at least one violation
- `violations`: vector of `BondViolation` structs with full detail

Tolerance: a bond with expected length `L` passes if `tol_low*L <= actual <= tol_high*L`.
Default is 90%-110%.

Also checks inter-residue peptide bonds (C-N) between consecutive residues on
the same chain.
"""
function check_bond_lengths(result::Dict; batch::Int=1, tol_low::Float64=0.9, tol_high::Float64=1.1)
    feats = result["feats"]
    coords = result["coords"]
    return check_bond_lengths(feats, coords; batch=batch, tol_low=tol_low, tol_high=tol_high)
end

function check_bond_lengths(feats::Dict, coords; batch::Int=1, tol_low::Float64=0.9, tol_high::Float64=1.1)
    # Collect atom positions per residue
    coords_b = ndims(coords) == 2 ? coords : coords[:, :, batch]
    atom_pad_mask = feats["atom_pad_mask"][:, batch]
    atom_to_token = feats["atom_to_token"][:, :, batch]
    ref_atom_name_chars = feats["ref_atom_name_chars"][:, :, :, batch]
    res_type, residue_index, asym_id, mol_type, token_pad_mask, res_offset = _token_metadata(feats, batch)

    T_tokens = size(atom_to_token, 2)

    # Build per-token atom coordinate maps (protein residues only)
    token_atoms = [Dict{String, NTuple{3,Float32}}() for _ in 1:T_tokens]
    token_is_protein = falses(T_tokens)
    token_res_name = fill("", T_tokens)
    token_chain_id = fill("", T_tokens)
    token_res_seq = zeros(Int, T_tokens)

    for t in 1:T_tokens
        token_pad_mask[t] > 0.5 || continue
        Int(mol_type[t]) == chain_type_ids["PROTEIN"] || continue
        token_is_protein[t] = true
        token_res_name[t] = _res_name_from_onehot(view(res_type, :, t))
        token_chain_id[t] = string(chain_id_from_asym(Int(asym_id[t])))
        token_res_seq[t] = Int(residue_index[t]) + res_offset
    end

    for m in 1:size(coords_b, 2)
        atom_pad_mask[m] > 0.5 || continue
        token_idx = argmax(view(atom_to_token, m, :))
        token_is_protein[token_idx] || continue

        atom_name = decode_atom_name_chars(view(ref_atom_name_chars, :, :, m))
        _is_fake_or_mask_atom(atom_name) && continue

        x = coords_b[1, m]
        y = coords_b[2, m]
        z = coords_b[3, m]
        (isfinite(x) && isfinite(y) && isfinite(z)) || continue

        token_atoms[token_idx][atom_name] = (Float32(x), Float32(y), Float32(z))
    end

    # Check bonds
    violations = BondViolation[]
    n_checked = 0

    for t in 1:T_tokens
        token_is_protein[t] || continue
        amap = token_atoms[t]
        isempty(amap) && continue
        res = token_res_name[t]

        # Intra-residue bonds (backbone + sidechain)
        all_bonds = Tuple{String,String,Bool}[]
        for (a1, a2) in _BACKBONE_BONDS
            push!(all_bonds, (a1, a2, true))
        end
        sc = get(_SIDECHAIN_BONDS, res, Tuple{String,String}[])
        for (a1, a2) in sc
            push!(all_bonds, (a1, a2, false))
        end

        for (a1, a2, is_bb) in all_bonds
            haskey(amap, a1) && haskey(amap, a2) || continue

            # Get expected length (sp2 overrides take priority)
            expected = get(_SP2_BOND_LENGTH_OVERRIDES, (res, a1, a2), nothing)
            if expected === nothing
                expected = get(_EXPECTED_BOND_LENGTHS, (res, a1, a2), nothing)
            end
            if expected === nothing
                # Try generic backbone from any residue that has it
                for fallback_res in ("ALA", "GLY")
                    expected = get(_EXPECTED_BOND_LENGTHS, (fallback_res, a1, a2), nothing)
                    expected !== nothing && break
                end
            end
            expected === nothing && continue

            p1 = amap[a1]
            p2 = amap[a2]
            dist = sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2 + (p1[3]-p2[3])^2)

            lo = Float32(tol_low) * expected
            hi = Float32(tol_high) * expected
            n_checked += 1

            if dist < lo
                push!(violations, BondViolation(
                    token_res_seq[t], res, token_chain_id[t],
                    a1, a2, expected, dist, lo, hi, :too_short, is_bb,
                ))
            elseif dist > hi
                push!(violations, BondViolation(
                    token_res_seq[t], res, token_chain_id[t],
                    a1, a2, expected, dist, lo, hi, :too_long, is_bb,
                ))
            end
        end

        # Inter-residue peptide bond: C(t) -> N(t+1) on same chain
        if t < T_tokens && token_is_protein[t+1] && token_chain_id[t] == token_chain_id[t+1]
            amap_next = token_atoms[t+1]
            if haskey(amap, "C") && haskey(amap_next, "N")
                p1 = amap["C"]
                p2 = amap_next["N"]
                dist = sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2 + (p1[3]-p2[3])^2)

                expected = _PEPTIDE_BOND_LENGTH
                lo = Float32(tol_low) * expected
                hi = Float32(tol_high) * expected
                n_checked += 1

                if dist < lo
                    push!(violations, BondViolation(
                        token_res_seq[t], res, token_chain_id[t],
                        "C", "N+1", expected, dist, lo, hi, :too_short, true,
                    ))
                elseif dist > hi
                    push!(violations, BondViolation(
                        token_res_seq[t], res, token_chain_id[t],
                        "C", "N+1", expected, dist, lo, hi, :too_long, true,
                    ))
                end
            end
        end
    end

    # Count unique residues with violations
    violated_residues = Set{Tuple{String,Int}}()
    for v in violations
        push!(violated_residues, (v.chain_id, v.res_idx))
    end

    return (
        n_bonds_checked = n_checked,
        n_violations = length(violations),
        n_residues_with_violations = length(violated_residues),
        violations = violations,
    )
end

"""
    print_bond_length_report(result; batch=1, tol_low=0.9, tol_high=1.1, io=stdout)

Run bond length checks and print a human-readable report.
"""
function print_bond_length_report(result; batch::Int=1, tol_low::Float64=0.9, tol_high::Float64=1.1, io::IO=stdout)
    stats = check_bond_lengths(result; batch=batch, tol_low=tol_low, tol_high=tol_high)
    print_bond_length_report(stats; io=io)
end

function print_bond_length_report(stats::NamedTuple; io::IO=stdout)
    println(io, "  Bond length check: $(stats.n_bonds_checked) bonds, $(stats.n_violations) violations in $(stats.n_residues_with_violations) residues")
    if stats.n_violations == 0
        return stats
    end

    # Group violations by residue
    by_residue = Dict{Tuple{String,Int,String}, Vector{BondViolation}}()
    for v in stats.violations
        key = (v.chain_id, v.res_idx, v.res_name)
        push!(get!(by_residue, key, BondViolation[]), v)
    end

    # Count by AA type
    aa_counts = Dict{String,Int}()
    for v in stats.violations
        aa_counts[v.res_name] = get(aa_counts, v.res_name, 0) + 1
    end

    # Count backbone vs sidechain
    n_bb = count(v -> v.is_backbone, stats.violations)
    n_sc = stats.n_violations - n_bb

    println(io, "    backbone: $n_bb, sidechain: $n_sc")
    println(io, "    by AA type: ", join(["$aa=$n" for (aa, n) in sort(collect(aa_counts); by=last, rev=true)], ", "))

    # List each violated residue
    for key in sort(collect(keys(by_residue)); by=k->(k[1], k[2]))
        chain, idx, res = key
        vlist = by_residue[key]
        bonds_str = join(["$(v.atom1)-$(v.atom2)($(round(v.actual; digits=3))/$(round(v.expected; digits=3))=$(v.kind))" for v in vlist], ", ")
        println(io, "    $chain:$res$idx: $bonds_str")
    end

    return stats
end
