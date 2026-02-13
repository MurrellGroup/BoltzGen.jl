import Pkg

if !isdefined(@__MODULE__, :WORKSPACE_ROOT)
    const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
end

if !isdefined(@__MODULE__, :RUNFROMHERE_PROJECT)
    const RUNFROMHERE_PROJECT = joinpath(WORKSPACE_ROOT, "runfromhere")
end

project_toml = joinpath(RUNFROMHERE_PROJECT, "Project.toml")
isfile(project_toml) || error(
    "Missing run environment: $project_toml. " *
    "Create /Users/benmurrell/JuliaM3/BoltzGenJuliaPort/runfromhere/Project.toml.",
)

Pkg.activate(RUNFROMHERE_PROJECT)
