@testset "context" begin

@test ctx == CuCurrentContext()

let ctx2 = CuContext(dev)
    @test ctx2 == CuCurrentContext()    # ctor implicitly pushes
    activate(ctx)
    @test ctx == CuCurrentContext()

    @test device(ctx2) == dev

    CUDAdrv.unsafe_destroy!(ctx2)
end

let global_ctx2 = nothing
    CuContext(dev) do ctx2
        @test ctx2 == CuCurrentContext()
        @test ctx != ctx2
        global_ctx2 = ctx2
    end
    @test !CUDAdrv.isvalid(global_ctx2)
    @test ctx == CuCurrentContext()

    @test device(ctx) == dev
    @test device() == dev
    synchronize()
end

end

@testset "primary context" begin

pctx = CuPrimaryContext(dev)

@test !isactive(pctx)
unsafe_reset!(pctx)
@test !isactive(pctx)

@test flags(pctx) == 0
setflags!(pctx, CUDAdrv.CTX_SCHED_BLOCKING_SYNC)
@test flags(pctx) == CUDAdrv.CTX_SCHED_BLOCKING_SYNC

let global_ctx = nothing
    CuContext(pctx) do ctx
        @test CUDAdrv.isvalid(ctx)
        @test isactive(pctx)
        global_ctx = ctx
    end
    @test !isactive(pctx)
    @test !CUDAdrv.isvalid(global_ctx)
end

CuContext(pctx) do ctx
    @test CUDAdrv.isvalid(ctx)
    @test isactive(pctx)

    unsafe_reset!(pctx)

    @test !isactive(pctx)
    @test !CUDAdrv.isvalid(ctx)
end

let
    @test !isactive(pctx)

    ctx1 = CuContext(pctx)
    @test isactive(pctx)
    @test CUDAdrv.isvalid(ctx1)

    unsafe_reset!(pctx)
    @test !isactive(pctx)
    @test !CUDAdrv.isvalid(ctx1)
    CUDAdrv.valid_contexts

    ctx2 = CuContext(pctx)
    @test isactive(pctx)
    @test !CUDAdrv.isvalid(ctx1)
    @test CUDAdrv.isvalid(ctx2)

    unsafe_reset!(pctx)
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
