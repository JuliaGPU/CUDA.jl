using CUDA.CUSPARSE
using SparseArrays
using SparseArrays: nonzeros, nnz, rowvals
using CUDA.GPUArrays: GPUSparseDeviceVector, GPUSparseDeviceMatrixCSC, GPUSparseDeviceMatrixCSR,
                      GPUSparseDeviceMatrixBSR, GPUSparseDeviceMatrixCOO

@testset "cudaconvert" begin
    @test isbitstype(GPUSparseDeviceVector{Float32, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float32, AS.Global}, AS.Global})
    @test isbitstype(GPUSparseDeviceMatrixCSC{Float32, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float32, AS.Global}, AS.Global})
    @test isbitstype(GPUSparseDeviceMatrixCSR{Float32, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float32, AS.Global}, AS.Global})
    @test isbitstype(GPUSparseDeviceMatrixBSR{Float32, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float32, AS.Global}, AS.Global})
    @test isbitstype(GPUSparseDeviceMatrixCOO{Float32, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float32, AS.Global}, AS.Global})

    V = sprand(10, 0.5)
    cuV = CuSparseVector(V)
    @test cudaconvert(cuV) isa GPUSparseDeviceVector{Float64, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float64, AS.Global}, AS.Global}

    A = sprand(10, 10, 0.5)
    cuA = CuSparseMatrixCSC(A)
    @test cudaconvert(cuA) isa GPUSparseDeviceMatrixCSC{Float64, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float64, AS.Global}, AS.Global}

    cuA = CuSparseMatrixCSR(A)
    @test cudaconvert(cuA) isa GPUSparseDeviceMatrixCSR{Float64, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float64, AS.Global}, AS.Global}

    cuA = CuSparseMatrixCOO(A)
    @test cudaconvert(cuA) isa GPUSparseDeviceMatrixCOO{Float64, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float64, AS.Global}, AS.Global}

    cuA = CuSparseMatrixBSR(A, 2)
    @test cudaconvert(cuA) isa GPUSparseDeviceMatrixBSR{Float64, Cint, CuDeviceVector{Cint, AS.Global}, CuDeviceVector{Float64, AS.Global}, AS.Global}
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
                v = CUDA.reduce_warp(+, v)

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
