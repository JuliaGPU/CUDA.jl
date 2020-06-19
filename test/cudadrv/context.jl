@testset "context" begin

ctx = CuCurrentContext()
dev = device()

let ctx2 = CuContext(dev)
    @test ctx2 == CuCurrentContext()    # ctor implicitly pushes
    activate(ctx)
    @test ctx == CuCurrentContext()

    @test device(ctx2) == dev

    CUDA.unsafe_destroy!(ctx2)
end

let global_ctx2 = nothing
    CuContext(dev) do ctx2
        @test ctx2 == CuCurrentContext()
        @test ctx != ctx2
        global_ctx2 = ctx2
    end
    @test !CUDA.isvalid(global_ctx2)
    @test ctx == CuCurrentContext()

    @test device(ctx) == dev
    @test device() == dev
    synchronize()
end

end

@testset "primary context" begin

# FIXME: these trample over our globally-managed context

# pctx = CuPrimaryContext(device())

# @test !isactive(pctx)
# unsafe_reset!(pctx)
# @test !isactive(pctx)

# @test flags(pctx) == 0
# setflags!(pctx, CUDA.CTX_SCHED_BLOCKING_SYNC)
# @test flags(pctx) == CUDA.CTX_SCHED_BLOCKING_SYNC

# let global_ctx = nothing
#     CuContext(pctx) do ctx
#         @test CUDA.isvalid(ctx)
#         @test isactive(pctx)
#         global_ctx = ctx
#     end
#     @test !isactive(pctx)
#     @test !CUDA.isvalid(global_ctx)
# end

# CuContext(pctx) do ctx
#     @test CUDA.isvalid(ctx)
#     @test isactive(pctx)

#     unsafe_reset!(pctx)

#     @test !isactive(pctx)
#     @test !CUDA.isvalid(ctx)
# end

# let
#     @test !isactive(pctx)

#     ctx1 = CuContext(pctx)
#     @test isactive(pctx)
#     @test CUDA.isvalid(ctx1)

#     unsafe_reset!(pctx)
#     @test !isactive(pctx)
#     @test !CUDA.isvalid(ctx1)
#     CUDA.valid_contexts

#     ctx2 = CuContext(pctx)
#     @test isactive(pctx)
#     @test !CUDA.isvalid(ctx1)
#     @test CUDA.isvalid(ctx2)

#     unsafe_reset!(pctx)
# end

end


@testset "cache config" begin

config = cache_config()

cache_config!(CUDA.FUNC_CACHE_PREFER_L1)
@test cache_config() == CUDA.FUNC_CACHE_PREFER_L1

cache_config!(config)

end


@testset "shmem config" begin

config = shmem_config()

shmem_config!(CUDA.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE)
@test shmem_config() == CUDA.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE

shmem_config!(config)

end


@testset "limits" begin

lim = limit(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

lim += 1
limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, lim)
@test lim == limit(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, lim)

end
