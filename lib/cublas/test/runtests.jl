using cuBLAS
using ParallelTestRunner

const init_code = quote
    include(joinpath(@__DIR__, "setup.jl"))
end

testsuite = find_tests(@__DIR__)
delete!(testsuite, "setup")

runtests(cuBLAS, ARGS; init_code, testsuite)
