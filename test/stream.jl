@testset "stream" begin

let
    s = CuStream()
    synchronize(s)
    @test CUDAdrv.query(s) == true

    let s2 = CuStream()
        @test s != s2
        @test !(s == s2)
    end

    let s3 = CuStream(; flags=CUDAdrv.STREAM_NON_BLOCKING)
        @test s != s3
        @test !(s == s3)
    end

    prio = priority_range()
    let s = CuStream(; priority=first(prio))
        @test priority(s) == first(prio)
    end
    let s = CuStream(; priority=last(prio))
        @test priority(s) == last(prio)
    end

    synchronize(CuDefaultStream())
end

end
