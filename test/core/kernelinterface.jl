import KernelInterface
import KernelInterface as KI
using CUDACore

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(CUDABackend, "CUDACore", CUDACore, CuArray, CUDACore.CuDeviceArray)

function ki_subgroup_kernel(num, id, lane)
    l = KI.get_local_id()
    s = KI.get_local_size()
    i = l.x + (l.y - 1) * s.x
    @inbounds begin
        num[i] = KI.get_num_sub_groups()
        id[i] = KI.get_sub_group_id()
        lane[i] = KI.get_sub_group_local_id()
    end
    return
end

@testset "partial sub-groups" begin
    # a 33x2 workgroup is made up of 3 warps, the last one only partially filled
    workgroupsize = (33, 2)
    n = prod(workgroupsize)
    num = CuArray{UInt32}(undef, n)
    id = CuArray{UInt32}(undef, n)
    lane = CuArray{UInt32}(undef, n)
    KI.@kernel CUDABackend() workgroupsize=workgroupsize ki_subgroup_kernel(num, id, lane)
    @test all(==(3), Array(num))
    @test Array(id) == [div(i, 32) + 1 for i in 0:n-1]
    @test Array(lane) == [rem(i, 32) + 1 for i in 0:n-1]
end

@testset "versioninfo" begin
    @test occursin("CUDA toolchain", sprint(KI.versioninfo, CUDABackend()))
end
