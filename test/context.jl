@testset "context" begin

@test ctx == CuCurrentContext()
@test ctx === CuCurrentContext()

@test_throws ErrorException deepcopy(ctx)

let ctx2 = CuContext(dev)
    @test ctx2 == CuCurrentContext()    # ctor implicitly pushes
    activate(ctx)
    @test ctx == CuCurrentContext()

    @test_throws ErrorException device(ctx2)

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
synchronize(ctx)
synchronize()

end

@testset "primary context" begin

pctx = CuPrimaryContext(dev)

@test !isactive(pctx)

@test flags(pctx) == CUDAdrv.SCHED_AUTO
setflags!(pctx, CUDAdrv.SCHED_BLOCKING_SYNC)
@test flags(pctx) == CUDAdrv.SCHED_BLOCKING_SYNC

CuContext(pctx) do ctx
    @test CUDAdrv.isvalid(ctx)
    @test isactive(pctx)
end
gc()
@test !isactive(pctx)

CuContext(pctx) do ctx
    @test CUDAdrv.isvalid(ctx)
    @test isactive(pctx)

    unsafe_reset!(pctx)

    @test !isactive(pctx)
    @test !CUDAdrv.isvalid(ctx)
end

end
