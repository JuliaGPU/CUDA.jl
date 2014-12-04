using CUDA, Base.Test

PERFORMANCE = haskey(ENV, "PERFORMANCE")
CODESPEED = get(ENV, "CODESPEED", nothing)
if CODESPEED != nothing && !PERFORMANCE
    error("Cannot submit to Codespeed without enabling performance measurements")
end

include("perfutil.jl")

@test devcount() > 0
include("core.jl")

dev = CuDevice(0)
if capability(dev) < v"2.0"
    warn("native execution not supported on SM < 2.0")
else
    include("native.jl")
end

include("bugs.jl")
