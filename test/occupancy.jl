@testset "occupancy" begin

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
    dummy = CuFunction(md, "dummy")

    active_blocks(dummy, 1)
    active_blocks(dummy, 1; shmem=64)

    occupancy(dummy, 1)
    occupancy(dummy, 1; shmem=64)

    launch_configuration(dummy)
    launch_configuration(dummy; shmem=64)
    launch_configuration(dummy; shmem=64, max_threads=64)

    cb_calls = 0
    launch_configuration(dummy; shmem=threads->(cb_calls += 1; 0))
    @test cb_calls > 0
end

end
