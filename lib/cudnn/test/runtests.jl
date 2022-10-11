using Distributed, ReTest

# add workers (passing along crucial command-line flags)
const exeflags = Base.julia_cmd()
filter!(exeflags.exec) do c
    return !(startswith(c, "--depwarn") || startswith(c, "--check-bounds"))
end
push!(exeflags.exec, "--check-bounds=yes")
push!(exeflags.exec, "--startup-file=no")
push!(exeflags.exec, "--depwarn=yes")
push!(exeflags.exec, "--project=$(Base.active_project())")
const exename = popfirst!(exeflags.exec)
function addworker(X; kwargs...)
    withenv("JULIA_NUM_THREADS" => 1, "OPENBLAS_NUM_THREADS" => 1) do
        addprocs(X; exename, exeflags, kwargs...)
    end
end
addworker(Threads.nthreads())

# include all tests
@everywhere include("tests.jl")

