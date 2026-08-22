import KernelInterface
import KernelInterface as KI
using CUDACore

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(CUDABackend(), CuArray)

function ki_subgroup_kernel(num, sizes, id, lane)
    l = KI.get_local_id()
    s = KI.get_local_size()
    i = l.x + (l.y - 1) * s.x
    @inbounds begin
        num[i] = KI.get_num_sub_groups()
        sizes[i] = KI.get_sub_group_size()
        id[i] = KI.get_sub_group_id()
        lane[i] = KI.get_sub_group_local_id()
    end
    return
end

# KernelInterface leaves the formation of sub-groups unspecified; CUDA forms warps from
# consecutive linear thread indices
@testset "partial sub-groups" begin
    # a 33x2 workgroup is made up of 3 warps, the last one only partially filled
    workgroupsize = (33, 2)
    n = prod(workgroupsize)
    num = CuArray{UInt32}(undef, n)
    sizes = CuArray{UInt32}(undef, n)
    id = CuArray{UInt32}(undef, n)
    lane = CuArray{UInt32}(undef, n)
    KI.@launch CUDABackend() workgroupsize=workgroupsize ki_subgroup_kernel(num, sizes, id, lane)
    @test all(==(3), Array(num))
    @test Array(sizes) == [i < 64 ? 32 : 2 for i in 0:n-1]
    @test Array(id) == [div(i, 32) + 1 for i in 0:n-1]
    @test Array(lane) == [rem(i, 32) + 1 for i in 0:n-1]
end

@testset "copyto!" begin
    backend = CUDABackend()

    # bits unions store their type tags separately
    host = Union{Missing, Int32}[1, missing, 3]
    dev = CuArray{Union{Missing, Int32}}(undef, 3)
    @test KI.copyto!(backend, dev, host) === dev
    back = Vector{Union{Missing, Int32}}(undef, 3)
    KI.copyto!(backend, back, dev)
    KI.synchronize(backend)
    @test isequal(back, host)

    # host to host
    a = zeros(Float32, 4)
    @test KI.copyto!(backend, a, ones(Float32, 4)) === a
    @test a == ones(Float32, 4)

    # only dense arrays
    @test_throws ArgumentError KI.copyto!(backend, view(CUDA.zeros(Float32, 8), 1:2:8), CUDA.ones(Float32, 4))
end

@testset "versioninfo" begin
    @test occursin("CUDA toolchain", sprint(KI.versioninfo, CUDABackend()))
end
