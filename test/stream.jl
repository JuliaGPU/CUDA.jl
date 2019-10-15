@testset "stream" begin

let
    s = CuStream()
    synchronize(s)
    let s2 = CuStream()
        @test s != s2
    end

    let s3 = CuStream(CUDAdrv.CU_STREAM_NON_BLOCKING)
        @test s != s3
    end

    synchronize(CuDefaultStream())
end

end
