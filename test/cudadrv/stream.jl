@testset "stream" begin

@testcase "essentials" begin
    s = CuStream()
    synchronize(s)
    @test CUDA.query(s) == true

    s2 = CuStream()
        @test s != s2
    @test !(s == s2)

    s3 = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    @test s != s3
    @test !(s == s3)
end

@testcase "priorities" begin
    prio = priority_range()

    let s = CuStream(; priority=first(prio))
        @test priority(s) == first(prio)
    end

    let s = CuStream(; priority=last(prio))
        @test priority(s) == last(prio)
    end
end

@testcase "synchronization" begin
    synchronize()
    synchronize(; blocking=false)
    synchronize(stream())
    synchronize(stream(); blocking=false)
end

end
