@testset "thread block clusters" begin

# the barriers must survive to device IR on every Julia; on LLVM <= 16 they
# used to demote to a silent trap stub (runs on any hardware)
@testset "barrier codegen" begin
    @test @filecheck CUDA.code_llvm(Tuple{}) do
        @check "llvm.nvvm.barrier.cluster.arrive"
        @check "llvm.nvvm.barrier.cluster.wait"
        @check_not "gpu_report_exception"
        cluster_arrive()
        cluster_wait()
    end
end

if capability(device()) >= v"9.0" && VERSION >= v"1.11-"

###########################################################################################

@testset "indexing" begin
    function f(A::AbstractArray{Int32,9})
        ti = threadIdx().x
        tj = threadIdx().y
        tk = threadIdx().z
        bi = blockIdxInCluster().x
        bj = blockIdxInCluster().y
        bk = blockIdxInCluster().z
        ci = clusterIdx().x
        cj = clusterIdx().y
        ck = clusterIdx().z
        A[ti,tj,tk,bi,bj,bk,ci,cj,ck] = 1
        nothing
    end

    threads = (3,5,7)
    clustersize = (2,2,2)
    blocks = (4,6,8)
    A = CUDA.zeros(Int32, threads..., clustersize..., (blocks .÷ clustersize)...)
    @cuda threads=threads blocks=blocks clustersize=clustersize f(A)

    @test all(==(1), Array(A))
end

###########################################################################################

@testset "cluster dimensions and linear indices" begin
    function f(A::AbstractArray{Int32,2})
        b = blockIdx().x + (blockIdx().y - 1i32) * gridDim().x +
            (blockIdx().z - 1i32) * gridDim().x * gridDim().y
        A[1,b] = clusterDim().x
        A[2,b] = clusterDim().y
        A[3,b] = clusterDim().z
        A[4,b] = gridClusterDim().x
        A[5,b] = gridClusterDim().y
        A[6,b] = gridClusterDim().z
        A[7,b] = linearClusterSize()
        A[8,b] = linearBlockIdxInCluster()
        A[9,b] = blockIdxInCluster().x
        A[10,b] = blockIdxInCluster().y
        A[11,b] = blockIdxInCluster().z
        A[12,b] = clusterIdx().x
        A[13,b] = clusterIdx().y
        A[14,b] = clusterIdx().z
        nothing
    end

    clustersize = (2,2,2)
    blocks = (4,2,6)
    A = CUDA.zeros(Int32, 14, prod(blocks))
    @cuda threads=1 blocks=blocks clustersize=clustersize f(A)
    A = Array(A)

    nclusters = blocks .÷ clustersize
    for (b, I) in enumerate(CartesianIndices(blocks))
        bidx = Tuple(I)
        inc = mod1.(bidx, clustersize)
        @test A[1:3,b] == collect(clustersize)
        @test A[4:6,b] == collect(nclusters)
        @test A[7,b] == prod(clustersize)
        # the linear rank within the cluster is x-major
        @test A[8,b] == LinearIndices(clustersize)[inc...]
        @test A[9:11,b] == collect(inc)
        @test A[12:14,b] == collect(cld.(bidx, clustersize))
    end
end

###########################################################################################

@testset "distributed shared memory" begin
    function f(A::AbstractArray{Int32,3})
        ti = threadIdx().x
        nt = blockDim().x
        @assert 1<=ti<=nt
        bi = blockIdxInCluster().x
        nb = clusterDim().x
        @assert 1<=bi<=nb
        ci = clusterIdx().x
        nc = gridClusterDim().x
        @assert 1<=ci<=nc

        sm = CuStaticSharedArray(Int32, 8)
        for i in 1:nb
            sm[i] = -1
        end
        cluster_wait()

        for i in 1:nb
            dsm = CuDistributedSharedArray(sm, i)
            dsm[bi] = bi
        end
        cluster_wait()

        for i in 1:nb
            A[i,bi,ci] = sm[i]
        end
        return nothing
    end

    threads = 1
    clustersize = 4
    blocks = 16
    A = CUDA.zeros(Int32, clustersize, clustersize, blocks ÷ clustersize)
    @cuda threads=threads blocks=blocks clustersize=clustersize f(A)

    B = Array(A)
    goodB = [i for i in 1:clustersize, bi in 1:clustersize, ci in 1:blocks ÷ clustersize]
    @test B == goodB
end

###########################################################################################

end
end
