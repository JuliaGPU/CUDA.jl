@testset "events" begin

let
    start = CuEvent()
    stop = CuEvent()
    @test start != stop
    record(start)
    record(stop)
    synchronize(stop)
    @test elapsed(start, stop) > 0
    @test (CUDAdrv.@elapsed begin
        end) > 0
    @test (CUDAdrv.@elapsed CuDefaultStream() begin
        end) > 0
end

end
