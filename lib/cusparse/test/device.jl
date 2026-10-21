using cuSPARSE
using SparseArrays
using SparseArrays: nonzeros, nnz, rowvals
using GPUArrays: GPUSparseVector,GPUSparseMatrixCSC,GPUSparseMatrixCSR,GPUSparseMatrixCOO

@testset "cudaconvert" begin
    V=sprand(10,0.5)
    for (host,constructor,format) in ((V,CuSparseVector,GPUSparseVector),
        (sprand(10,10,0.5),CuSparseMatrixCSC,GPUSparseMatrixCSC),
        (sprand(10,10,0.5),CuSparseMatrixCSR,GPUSparseMatrixCSR),
        (sprand(10,10,0.5),CuSparseMatrixCOO,GPUSparseMatrixCOO))
        dev=cudaconvert(constructor(host))
        @test dev isa format
        @test isbitstype(typeof(dev))
        @test nonzeros(dev) isa CuDeviceVector{Float64,AS.Global}
    end
    bsr=cudaconvert(CuSparseMatrixBSR(sprand(10,10,0.5),2))
    @test bsr isa cuSPARSE.CuSparseDeviceMatrixBSR
    @test isbitstype(typeof(bsr))
end

@testset "device SparseArrays api" begin
    @testset "BSR stored scalar count" begin
        function count_bsr_kernel(out,A)
            out[1]=nnz(A)
            nothing
        end
        A=CuSparseMatrixBSR(sparse(Float32[1 0;0 2]),2)
        out=CuVector{Int}(undef,1)
        @cuda threads=1 count_bsr_kernel(out,A)
        @test only(Array(out))==nnz(A)==4
    end
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
