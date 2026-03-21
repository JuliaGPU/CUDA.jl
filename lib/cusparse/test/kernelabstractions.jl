import KernelAbstractions
import KernelAbstractions as KA
import Adapt: adapt

@testset "KernelAbstractions adapt" begin
    # CPU → GPU
    A = sprand(Float32, 10, 10, 0.5)
    A_d = adapt(CUDABackend(), A)
    @test A_d isa cuSPARSE.CuSparseMatrixCSC
    @test adapt(CUDABackend(), A_d) |> typeof == typeof(A_d)

    # GPU → CPU
    B_d = A |> cu
    B = adapt(KA.CPU(), A_d)
    @test B isa SparseMatrixCSC
    @test adapt(KA.CPU(), B) |> typeof == typeof(B)
end

# run the KA sparse test that CUDA.jl's core tests skip
include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))
@testset "KernelAbstractions sparse" begin
    backend = CUDABackend(false, false)
    backendT = typeof(backend).name.wrapper
    A = KA.allocate(backend, Float32, 5, 5)
    @test @inferred(KA.get_backend(sparse(A))) isa backendT
end
