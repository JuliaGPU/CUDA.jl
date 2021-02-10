@testset "events" begin

@testcase "essentials" begin
    start = CuEvent()
    stop = CuEvent()
    @test start != stop

    CUDA.record(start)
    CUDA.record(stop)
    synchronize(stop)

    @test elapsed(start, stop) > 0
end

@testcase "@elapsed" begin
    @test (CUDA.@elapsed begin end) > 0
end

@testcase "flags" begin
    CuEvent(CUDA.EVENT_BLOCKING_SYNC)
    CuEvent(CUDA.EVENT_BLOCKING_SYNC | CUDA.EVENT_DISABLE_TIMING)
end

@testcase "stream wait" begin
    event  = CuEvent()
    stream = CuStream()

    CUDA.record(event, stream)

    CUDA.wait(event)
    synchronize()
end

@testcase "event query" begin
    event  = CuEvent()
    @test CUDA.query(event) == true
end

end
