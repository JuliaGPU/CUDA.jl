using CUDA.CUSPARSE
using SparseArrays
using CUDA.CUSPARSE: CuSparseDeviceVector, CuSparseDeviceMatrixCSC, CuSparseDeviceMatrixCSR,
                     CuSparseDeviceMatrixBSR, CuSparseDeviceMatrixCOO

@testset "cudaconvert" begin
    @test isbitstype(CuSparseDeviceVector{Float32, Cint, CUDA.AS.Global})
    @test isbitstype(CuSparseDeviceMatrixCSC{Float32, Cint, CUDA.AS.Global})
    @test isbitstype(CuSparseDeviceMatrixCSR{Float32, Cint, CUDA.AS.Global})
    @test isbitstype(CuSparseDeviceMatrixBSR{Float32, Cint, CUDA.AS.Global})
    @test isbitstype(CuSparseDeviceMatrixCOO{Float32, Cint, CUDA.AS.Global})

    V = sprand(10, 0.5)
    cuV = CuSparseVector(V)
    @test cudaconvert(cuV) isa CuSparseDeviceVector{Float64, Cint, 1}

    A = sprand(10, 10, 0.5)
    cuA = CuSparseMatrixCSC(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCSC{Float64, Cint, 1}

    cuA = CuSparseMatrixCSR(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCSR{Float64, Cint, 1}

    cuA = CuSparseMatrixCOO(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCOO{Float64, Cint, 1}

    # Roger-Luo: I'm not sure how to create a BSR matrix
    # cuA = CuSparseMatrixBSR(A)
    # @test cudaconvert(cuA) isa CuSparseDeviceMatrixBSR
end
