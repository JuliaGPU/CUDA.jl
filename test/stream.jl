@testset "stream" begin

let
    s = CuStream()
    synchronize(s)
    let s2 = CuStream()
        @test s != s2
    end

    synchronize(CuDefaultStream())
end

end
