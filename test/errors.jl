@testset "errors" begin

let
    ex = CuError(0)
    @test CUDAdrv.name(ex) == :SUCCESS
    @test CUDAdrv.description(ex) == "Success"
    
    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test contains(str, "0")
    @test contains(str, "Success")
end

let
    ex = CuError(0, "foobar")
    
    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test contains(str, "foobar")
end

end
