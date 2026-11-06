using cuSPARSE
using SparseArrays
using SparseArrays: nonzeros, nnz, rowvals
using CUDACore.GPUArrays: GPUSparseVector, GPUSparseMatrixCSC, GPUSparseMatrixCSR,
                          GPUSparseMatrixCOO

# the canonical formats are the same structs on the device, with device vectors
@testset "cudaconvert" begin
    DV{T} = CuDeviceVector{T, AS.Global}
    @test isbitstype(GPUSparseVector{Float32, Cint, DV{Cint}, DV{Float32}})
    @test isbitstype(GPUSparseMatrixCSC{Float32, Cint, DV{Cint}, DV{Float32}})
    @test isbitstype(GPUSparseMatrixCSR{Float32, Cint, DV{Cint}, DV{Float32}})
    @test isbitstype(GPUSparseMatrixCOO{Float32, Cint, DV{Cint}, DV{Float32}})
    @test isbitstype(cuSPARSE.CuSparseDeviceMatrixBSR{Float32, Cint, DV{Cint}, DV{Float32}})

    V = sprand(10, 0.5)
    cuV = CuSparseVector(V)
    @test cudaconvert(cuV) isa GPUSparseVector{Float64, Cint, DV{Cint}, DV{Float64}}

    A = sprand(10, 10, 0.5)
    cuA = CuSparseMatrixCSC{Float64,Cint}(A)
    @test cudaconvert(cuA) isa GPUSparseMatrixCSC{Float64, Cint, DV{Cint}, DV{Float64}}

    cuA = CuSparseMatrixCSR{Float64,Cint}(A)
    @test cudaconvert(cuA) isa GPUSparseMatrixCSR{Float64, Cint, DV{Cint}, DV{Float64}}

    cuA = CuSparseMatrixCOO{Float64,Cint}(A)
    @test cudaconvert(cuA) isa GPUSparseMatrixCOO{Float64, Cint, DV{Cint}, DV{Float64}}

    cuA = CuSparseMatrixBSR{Float64,Cint}(A, 2)
    @test cudaconvert(cuA) isa cuSPARSE.CuSparseDeviceMatrixBSR{Float64, Cint, DV{Cint}, DV{Float64}}
end

@testset "device SparseArrays api" begin
    @testset "nnz per column" begin
        function nnz_per_column(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
            function nnz_per_column_kernel(out, A)
                i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                col = @view A[:, i]
                out[i] = nnz(col)
                nothing
            end

            out = CuVector{Ti}(undef, size(A, 2))
            @cuda threads=size(A, 2) nnz_per_column_kernel(out, A)
            out
        end

        nnz_per_column(A::SparseMatrixCSC) = map(nnz, eachcol(A))

        A = sprand(10, 10, 0.5)
        cuA = CuSparseMatrixCSC(A)

        @test nnz_per_column(A) == Vector(nnz_per_column(cuA))
    end

    @testset "sum per column" begin
        function sum_per_column(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
            function sum_per_column_kernel(out, A)
                j = blockIdx().x
                col = @view A[:, j]
                
                v = zero(Tv)
                i = threadIdx().x
                while i <= nnz(col)
                    v += nonzeros(col)[i]
                    i += blockDim().x
                end
                v = CUDACore.reduce_warp(+, v)

                if threadIdx().x == 1
                    out[j] = v
                end
                nothing
            end

            out = CuVector{Tv}(undef, size(A, 2))
            @cuda threads=32 blocks=size(A, 2) sum_per_column_kernel(out, A)
            out
        end

        sum_per_column(A::SparseMatrixCSC) = vec(sum(A; dims=1))

        A = sprand(10, 10, 0.5)
        cuA = CuSparseMatrixCSC(A)

        @test sum_per_column(A) ≈ Vector(sum_per_column(cuA))
    end

    @testset "last nonzero per column" begin
        function last_nz_per_column(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
            function last_nz_per_column_kernel(out, A)
                i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                col = @view A[:, i]
                out[i] = last(rowvals(col))
                nothing
            end

            out = CuVector{Ti}(undef, size(A, 2))
            @cuda threads=size(A, 2) last_nz_per_column_kernel(out, A)
            out
        end

        last_nz_per_column(A::SparseMatrixCSC) = map(last ∘ SparseArrays.rowvals, eachcol(A))

        A = sprand(10, 10, 0.5)
        cuA = CuSparseMatrixCSC(A)

        @test last_nz_per_column(A) == Vector(last_nz_per_column(cuA))
    end
end
