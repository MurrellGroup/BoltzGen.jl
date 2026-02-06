import Pkg

const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(WORKSPACE_ROOT, "Onion.jl"))

function has_option(args::Vector{String}, name::AbstractString)
    needle = "--" * String(name)
    return any(a -> a == needle, args)
end

function option_value(args::Vector{String}, name::AbstractString, default::String)
    needle = "--" * String(name)
    i = findfirst(==(needle), args)
    if i === nothing
        return default
    end
    if i < length(args) && !startswith(args[i + 1], "--")
        return args[i + 1]
    end
    return "true"
end

forwarded = copy(ARGS)

has_option(forwarded, "target") || error(
    "Missing --target <path-to-pdb-or-cif>",
)

if !has_option(forwarded, "target-design-mask")
    # Folding mode keeps target residues fixed by default.
    push!(forwarded, "--target-design-mask")
    push!(forwarded, "")
end

if !has_option(forwarded, "design-length")
    # No appended design chain in fold mode unless explicitly requested.
    push!(forwarded, "--design-length")
    push!(forwarded, "0")
end

with_affinity = lowercase(option_value(forwarded, "with-affinity", "false")) == "true"
if !has_option(forwarded, "weights")
    default_weights = with_affinity ?
        joinpath(WORKSPACE_ROOT, "boltzgen_cache", "boltz2_aff_state_dict.safetensors") :
        joinpath(WORKSPACE_ROOT, "boltzgen_cache", "boltz2_conf_final_state_dict.safetensors")
    push!(forwarded, "--weights")
    push!(forwarded, default_weights)
end

if !has_option(forwarded, "with-confidence")
    push!(forwarded, "--with-confidence")
    push!(forwarded, "true")
end

if !has_option(forwarded, "include-nonpolymer")
    push!(forwarded, "--include-nonpolymer")
    push!(forwarded, "true")
end

empty!(ARGS)
append!(ARGS, forwarded)

include(normpath(joinpath(@__DIR__, "run_target_conditioned_design.jl")))

# main() is defined in run_target_conditioned_design.jl.
main()
