using CUDA.CUSPARSE
using SparseArrays
using CUDA.CUSPARSE: CuSparseDeviceVector, CuSparseDeviceMatrixCSC, CuSparseDeviceMatrixCSR,
                     CuSparseDeviceMatrixBSR, CuSparseDeviceMatrixCOO

@testset "cudaconvert" begin
    @test isbitstype(CuSparseDeviceVector{Float32, Cint, AS.Global})
    @test isbitstype(CuSparseDeviceMatrixCSC{Float32, Cint, AS.Global})
    @test isbitstype(CuSparseDeviceMatrixCSR{Float32, Cint, AS.Global})
    @test isbitstype(CuSparseDeviceMatrixBSR{Float32, Cint, AS.Global})
    @test isbitstype(CuSparseDeviceMatrixCOO{Float32, Cint, AS.Global})

    V = sprand(10, 0.5)
    cuV = CuSparseVector(V)
    @test cudaconvert(cuV) isa CuSparseDeviceVector{Float64, Cint, AS.Global}

    A = sprand(10, 10, 0.5)
    cuA = CuSparseMatrixCSC(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCSC{Float64, Cint, AS.Global}

    cuA = CuSparseMatrixCSR(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCSR{Float64, Cint, AS.Global}

    cuA = CuSparseMatrixCOO(A)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixCOO{Float64, Cint, AS.Global}

    cuA = CuSparseMatrixBSR(A, 2)
    @test cudaconvert(cuA) isa CuSparseDeviceMatrixBSR{Float64, Cint, AS.Global}
end

@testset "device SparseArrays api" begin
    @testset "nnz per column" begin
        function nnz_per_column(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
            function nnz_per_column_kernel(out, A)
                i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                col = @view A[:, i]
                out[i] = SparseArrays.nnz(col)
                nothing
            end

            out = CuVector{Ti}(undef, size(A, 2))
            @cuda threads=size(A, 2) nnz_per_column_kernel(out, A)
            out
        end

        nnz_per_column(A::SparseMatrixCSC) = map(SparseArrays.nnz, eachcol(A))

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
                while i <= SparseArrays.nnz(col)
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
                out[i] = last(SparseArrays.rowvals(col))
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