#!/usr/bin/env julia
#
# Build a real MSA for hen egg-white lysozyme (HEWL, 129 residues)
# from UniProt lysozyme family sequences.
#
# Strategy: use the mature HEWL sequence as query, align each homolog
# via Needleman-Wunsch, filter by coverage and identity, output A3M.
#

# ── Read FASTA ───────────────────────────────────────────────────────────────

function read_fasta(path::String)
    entries = Tuple{String,String}[]
    header = ""
    seq = IOBuffer()
    for line in readlines(path)
        line = strip(line)
        isempty(line) && continue
        if startswith(line, ">")
            if !isempty(header)
                push!(entries, (header, String(take!(seq))))
            end
            header = String(line[2:end])
        else
            print(seq, line)
        end
    end
    if !isempty(header) && position(seq) > 0
        push!(entries, (header, String(take!(seq))))
    end
    return entries
end

# ── Simple Needleman-Wunsch alignment ───────────────────────────────────────

function nw_align(query::String, target::String; match_score=2, mismatch=-1, gap=-2)
    m = length(query)
    n = length(target)
    # Score matrix
    H = zeros(Int, m+1, n+1)
    for i in 1:m; H[i+1, 1] = gap * i; end
    for j in 1:n; H[1, j+1] = gap * j; end
    for i in 1:m, j in 1:n
        s = query[i] == target[j] ? match_score : mismatch
        H[i+1, j+1] = max(H[i, j] + s, H[i+1, j] + gap, H[i, j+1] + gap)
    end

    # Traceback
    q_aligned = Char[]
    t_aligned = Char[]
    i, j = m, n
    while i > 0 || j > 0
        if i > 0 && j > 0
            s = query[i] == target[j] ? match_score : mismatch
            if H[i+1, j+1] == H[i, j] + s
                pushfirst!(q_aligned, query[i])
                pushfirst!(t_aligned, target[j])
                i -= 1; j -= 1
                continue
            end
        end
        if i > 0 && H[i+1, j+1] == H[i, j+1] + gap
            pushfirst!(q_aligned, query[i])
            pushfirst!(t_aligned, '-')
            i -= 1
        else
            pushfirst!(q_aligned, '-')
            pushfirst!(t_aligned, target[j])
            j -= 1
        end
    end

    return String(q_aligned), String(t_aligned)
end

# ── Build A3M from pairwise alignments ──────────────────────────────────────

function build_a3m(query_seq::String, entries::Vector{Tuple{String,String}};
                   max_rows::Int=100, min_identity::Float64=0.25, min_coverage::Float64=0.5)
    # For each entry, align to query and convert to A3M row
    # A3M format: positions matching query columns, lowercase for insertions, '-' for deletions
    rows = Tuple{Float64,String,String}[]  # (identity, header, a3m_row)

    for (header, seq) in entries
        # Strip signal peptides: only keep sequences in reasonable length range
        # HEWL mature is 129 residues
        length(seq) < 80 && continue
        length(seq) > 200 && continue

        q_aln, t_aln = nw_align(query_seq, seq)

        # Convert to A3M row: keep only columns where query has a residue
        a3m = Char[]
        n_match = 0
        n_ident = 0
        for k in eachindex(q_aln)
            if q_aln[k] != '-'
                # Query position: emit target char (or gap)
                push!(a3m, t_aln[k])
                n_match += 1
                if q_aln[k] == t_aln[k]
                    n_ident += 1
                end
            else
                # Insertion in target relative to query: lowercase in A3M
                if t_aln[k] != '-'
                    push!(a3m, lowercase(t_aln[k]))
                end
            end
        end

        # Ensure A3M row has exactly query_len uppercase/gap positions
        # Pad with gaps if short (can happen at sequence termini)
        n_aligned = count(c -> isuppercase(c) || c == '-', a3m)
        while n_aligned < length(query_seq)
            push!(a3m, '-')
            n_aligned += 1
        end

        identity = n_match > 0 ? n_ident / n_match : 0.0
        coverage = count(c -> c != '-', a3m) / length(query_seq)

        identity >= min_identity && coverage >= min_coverage || continue

        push!(rows, (identity, header, String(a3m)))
    end

    # Deduplicate by A3M sequence
    seen = Set{String}()
    unique_rows = Tuple{Float64,String,String}[]
    for r in rows
        if r[3] ∉ seen
            push!(seen, r[3])
            push!(unique_rows, r)
        end
    end

    # Sort by identity (highest first), take top max_rows
    sort!(unique_rows; by=x -> -x[1])
    selected = unique_rows[1:min(end, max_rows)]

    return selected
end

# ── Main ────────────────────────────────────────────────────────────────────

# Hen egg-white lysozyme mature sequence (129 residues, UniProt P00698)
const HEWL_MATURE = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"

println("Query: HEWL ($(length(HEWL_MATURE)) residues)")

# Read all downloaded sequences
entries = read_fasta("/tmp/lysozyme_combined.fasta")
println("Input sequences: $(length(entries))")

# Strip signal peptides: many lysozymes have ~18 residue signal peptides
# We align to the mature form, the aligner handles length differences

# Build A3M with 100 MSA rows
a3m_rows = build_a3m(HEWL_MATURE, entries; max_rows=100, min_identity=0.25, min_coverage=0.5)
println("MSA rows passing filters: $(length(a3m_rows))")

# Write A3M file
outpath = joinpath(@__DIR__, "..", "examples", "lysozyme_msa.a3m")
open(outpath, "w") do io
    println(io, ">query HEWL_P00698 $(length(HEWL_MATURE))aa")
    println(io, HEWL_MATURE)
    for (ident, header, row) in a3m_rows
        println(io, ">", header, " identity=", round(ident; digits=3))
        println(io, row)
    end
end

println("Written: $outpath")
println("Total MSA rows (including query): $(length(a3m_rows) + 1)")

# Verify: print identity distribution
identities = [r[1] for r in a3m_rows]
println("\nIdentity distribution:")
println("  min: ", round(minimum(identities); digits=3))
println("  max: ", round(maximum(identities); digits=3))
println("  mean: ", round(sum(identities)/length(identities); digits=3))

# Print first few rows for inspection
println("\nFirst 3 MSA rows:")
for (ident, header, row) in a3m_rows[1:min(3, end)]
    println("  ", round(ident; digits=2), "  ", row[1:min(60, end)], "...")
end
