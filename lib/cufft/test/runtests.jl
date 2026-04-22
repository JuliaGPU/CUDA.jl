using cuFFT
using ParallelTestRunner

const init_code = quote
    include(joinpath(@__DIR__, "setup.jl"))
end

testsuite = find_tests(@__DIR__)
delete!(testsuite, "setup")

runtests(cuFFT, ARGS; init_code, testsuite)
