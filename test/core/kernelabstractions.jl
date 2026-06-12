import KernelAbstractions
import KernelAbstractions as KA

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

ka_skip_tests = Set{String}(["sparse"])
Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDA, CuArray, CuDeviceArray; skip_tests=Set([
    "CPU synchronization",
    "fallback test: callable types",]))
for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
    Testsuite.unittest_testsuite(()->CUDABackend(PreferBlocks, AlwaysInline), "CUDA", CUDA, CuDeviceArray;
                                 skip_tests=ka_skip_tests)
end

@testset "KA.functional" begin
    @test KA.functional(CUDABackend()) == CUDA.functional()
end
