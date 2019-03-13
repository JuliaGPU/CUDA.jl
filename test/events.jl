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

@test (CUDAdrv.@elapsed begin
    end) > 0

@test (CUDAdrv.@elapsed CuDefaultStream() begin
    end) > 0

CuEvent(CUDAdrv.EVENT_BLOCKING_SYNC)
CuEvent(CUDAdrv.EVENT_BLOCKING_SYNC | CUDAdrv.EVENT_DISABLE_TIMING)

# Useful to synchronize work between streams
@testset "Stream Wait" begin
    event  = CuEvent(CUDAdrv.EVENT_DISABLE_TIMING)
    stream = CuStream()
    # Enqueue work on stream here
    CUDAdrv.record(event, stream)

    # Wait for work on CuDefaultStream
    wait(event)
    synchronize()
end

end
