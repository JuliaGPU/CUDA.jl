@testset "context" begin

@test ctx == CuCurrentContext()
@test ctx === CuCurrentContext()

@test_throws ErrorException deepcopy(ctx)

let ctx2 = CuContext(dev)
    @test ctx2 == CuCurrentContext()    # ctor implicitly pushes
    activate(ctx)
    @test ctx == CuCurrentContext()

    @test device(ctx2) == dev

    destroy!(ctx2)
end

instances = length(CUDAdrv.context_instances)
CuContext(dev) do ctx2
    @test length(CUDAdrv.context_instances) == instances+1
    @test ctx2 == CuCurrentContext()
    @test ctx != ctx2
end
@test length(CUDAdrv.context_instances) == instances
@test ctx == CuCurrentContext()

@test device(ctx) == dev
@test device() == dev
synchronize()

end

@testset "primary context" begin

pctx = CuPrimaryContext(dev)

@test !isactive(pctx)
unsafe_reset!(pctx)
@test !isactive(pctx)

@test flags(pctx) == CUDAdrv.CTX_SCHED_AUTO
setflags!(pctx, CUDAdrv.CTX_SCHED_BLOCKING_SYNC)
@test flags(pctx) == CUDAdrv.CTX_SCHED_BLOCKING_SYNC

CuContext(pctx) do ctx
    @test CUDAdrv.isvalid(ctx)
    @test isactive(pctx)
end
GC.gc()
@test !isactive(pctx)

CuContext(pctx) do ctx
    @test CUDAdrv.isvalid(ctx)
    @test isactive(pctx)

    unsafe_reset!(pctx)

    @test !isactive(pctx)
    @test !CUDAdrv.isvalid(ctx)
end

end


@testset "cache config" begin

cache_config!(CUDAdrv.FUNC_CACHE_PREFER_L1)
@test cache_config() == CUDAdrv.FUNC_CACHE_PREFER_L1

end


@testset "shmem config" begin

shmem_config!(CUDAdrv.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE)
@test shmem_config() == CUDAdrv.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE

end


@testset "limits" begin

lim = limit(CUDAdrv.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

lim += 1
limit!(CUDAdrv.LIMIT_DEV_RUNTIME_SYNC_DEPTH, lim)
@test lim == limit(CUDAdrv.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

end
