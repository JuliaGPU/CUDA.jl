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

end
end
