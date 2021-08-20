using Test
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using CUDA: CuSparseDeviceVector, CuSparseDeviceMatrixCSC, CuSparseDeviceMatrixCSR,
    CuSparseDeviceMatrixBSR, CuSparseDeviceMatrixCOO

@testset "cudaconvert" begin
    V = sprand(10, 0.5)
    cuV = CuSparseVector(V)
    @test cudaconvert(cuV) isa CuSparseDeviceVector

    A = sprand(10, 10, 0.5)
    cuA = CuSparseMatrixCSC(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCSC

    cuA = CuSparseMatrixCSR(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCSR

    cuA = CuSparseMatrixCOO(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCOO

    # Roger-Luo: I'm not sure how to create a BSR matrix
    # cuA = CuSparseMatrixBSR(A)
    # @test cudaconvert(cuA) isa CuSparseDeviceMatrixBSR
end
