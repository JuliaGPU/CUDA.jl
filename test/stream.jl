@testset "stream" begin

let
    s = CuStream()
    synchronize(s)
    @test CUDAdrv.query(s) == true

    let s2 = CuStream()
        @test s != s2
        @test !(s == s2)
    end

    let s3 = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
        @test s != s3
        @test !(s == s3)
    end

    synchronize(CuDefaultStream())
end

end
