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

@test (CUDA.@elapsed begin end) > 0

@test (CUDA.@elapsed CuDefaultStream() begin end) > 0

CuEvent(CUDA.EVENT_BLOCKING_SYNC)
CuEvent(CUDA.EVENT_BLOCKING_SYNC | CUDA.EVENT_DISABLE_TIMING)

@testset "stream wait" begin
    event  = CuEvent()
    stream = CuStream()

    CUDA.record(event, stream)

    CUDA.wait(event)
    synchronize()
end

@testset "event query" begin
    event  = CuEvent()
    @test CUDA.query(event) == true
end

@testset "elapsed stream" begin
    stream = CuStream()
    @test (CUDA.@elapsed stream begin end) > 0
end

end
