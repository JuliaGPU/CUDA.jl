@testset "thread block clusters" begin
if capability(device()) >= v"9.0"

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

    A = CUDA.zeros(Int32, threads..., clustersize..., (blocks .÷ clustersize)...)

    threads = (3,5,7)
    clustersize = (2,2,2)
    blocks = (4,6,8)
    @cuda threads=threads blocks=blocks clustersize=clustersize f(A)

    @test all(==(1), Array(A))
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

    A = CUDA.zeros(Int32, clustersize, clustersize, blocks ÷ clustersize)

    threads = 1
    clustersize = 4
    blocks = 16
    @cuda threads=threads blocks=blocks clustersize=clustersize f(A)

    B = Array(A)
    goodB = [i for i in 1:clustersize, bi in 1:clustersize, ci in 1:blocks ÷ clustersize]
    @test B == goodB
end

###########################################################################################

end
end
