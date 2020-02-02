@testset "events" begin

let
    start = CuEvent()
    stop = CuEvent()
    @test start != stop

    record(start)
    record(stop)
    synchronize(stop)

    @test elapsed(start, stop) > 0
end

@test (CUDAdrv.@elapsed begin end) > 0

@test (CUDAdrv.@elapsed CuDefaultStream() begin end) > 0

CuEvent(CUDAdrv.EVENT_BLOCKING_SYNC)
CuEvent(CUDAdrv.EVENT_BLOCKING_SYNC | CUDAdrv.EVENT_DISABLE_TIMING)

@testset "stream wait" begin
    event  = CuEvent()
    stream = CuStream()

    CUDAdrv.record(event, stream)

    CUDAdrv.wait(event)
    synchronize()
end

@testset "event query" begin
    event  = CuEvent()
    @test CUDAdrv.query(event) == true
end

@testset "elapsed stream" begin
    stream = CuStream()
    @test (CUDAdrv.@elapsed stream begin end) > 0
end

end
