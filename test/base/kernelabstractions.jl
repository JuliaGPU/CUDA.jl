import KernelAbstractions
import KernelAbstractions as KA
using SparseArrays

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDA, CuArray, CuDeviceArray)
for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
    Testsuite.unittest_testsuite(()->CUDABackend(PreferBlocks, AlwaysInline), "CUDA", CUDA, CuDeviceArray)
end

@testset "KA.functional" begin
    @test KA.functional(CUDABackend()) == CUDA.functional()
end

@testset "CUDA Backend Adapt Tests" begin
    # CPU → GPU
    A = sprand(Float32, 10, 10, 0.5) #CSC
    A_d = adapt(CUDABackend(), A) 
    @test A_d isa CUSPARSE.CuSparseMatrixCSC
    @test adapt(CUDABackend(), A_d) |> typeof == typeof(A_d)

    # GPU → CPU
    B_d = A |> cu # CuCSC
    B = adapt(KA.CPU(), A_d)
    @test B isa SparseMatrixCSC
    @test adapt(KA.CPU(), B) |> typeof == typeof(B) 
end
