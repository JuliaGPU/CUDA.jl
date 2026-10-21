using GPUArrays

@testset "v13 legacy ABI preflight" begin
    host=sparse(Float32[2 0; 1 3])
    for S in (CuSparseMatrixCSR,CuSparseMatrixCSC), mixed in (true,false)
        base=S(host)
        p,i,v=GPUArrays.sparse_buffers(base)
        A=S(CuVector{Int64}(p),mixed ? copy(i) : CuVector{Int64}(i),copy(v),size(base))
        before=map(Array,GPUArrays.sparse_buffers(A))
        @test_throws ArgumentError ic02!(A)
        @test_throws ArgumentError ilu02!(A)
        @test_throws ArgumentError cuSPARSE.color(A,'O')
        @test map(Array,GPUArrays.sparse_buffers(A))==before
    end
end

@testset "v13 accumulation precision selection" begin
    for S in (CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseMatrixCOO)
        A=S(sparse(reshape(Float64[16777217,-16777216],1,2)))
        x=CuVector(Float64[1,1]); B=reshape(x,2,1)
        for policy in (GPUArrays.Portable(),GPUArrays.Automatic())
            y=CUDACore.zeros(Float32,1)
            GPUArrays.spmv!(y,'N',A,x,1f0,0f0;algorithm=policy)
            @test Array(y)==Float32[1]
            C=CUDACore.zeros(Float32,1,1)
            GPUArrays.spmm!(C,'N','N',A,B,1f0,0f0;algorithm=policy)
            @test Array(C)==reshape(Float32[1],1,1)
        end
        y=CuVector(Float32[7])
        @test_throws ArgumentError GPUArrays.spmv!(y,'N',A,x,1f0,0f0;algorithm=GPUArrays.Vendor())
        @test Array(y)==Float32[7]
    end
end

@testset "v13 half output does not narrow wider inputs" begin
    for S in (CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseMatrixCOO)
        A=S(sparse(reshape(Float32[2049,-2048],1,2)))
        x=CuVector(Float32[1,1]); B=reshape(x,2,1)
        for policy in (GPUArrays.Portable(),GPUArrays.Automatic())
            y=CUDACore.zeros(Float16,1)
            GPUArrays.spmv!(y,'N',A,x,Float16(1),Float16(0);algorithm=policy)
            @test Array(y)==Float16[1]
            C=CUDACore.zeros(Float16,1,1)
            GPUArrays.spmm!(C,'N','N',A,B,Float16(1),Float16(0);algorithm=policy)
            @test Array(C)==reshape(Float16[1],1,1)
        end
        y=CuVector(Float16[7])
        @test_throws ArgumentError GPUArrays.spmv!(y,'N',A,x,Float16(1),Float16(0);algorithm=GPUArrays.Vendor())
        @test Array(y)==Float16[7]
    end
end

@testset "v13 BSR component overlap" begin
    A=CuSparseMatrixBSR(sparse(Float32[2 0; 0 3]),1)
    for y in (A.nzVal,reinterpret(Float32,A.colVal),view(reinterpret(Float32,A.rowPtr),1:2))
        before=map(Array,(A.rowPtr,A.colVal,A.nzVal))
        @test_throws ArgumentError GPUArrays.spmv!(y,'N',A,CuVector(Float32[1,1]),1f0,0f0)
        @test map(Array,(A.rowPtr,A.colVal,A.nzVal))==before
    end
end
