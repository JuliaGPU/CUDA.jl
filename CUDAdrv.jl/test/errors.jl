@testset "errors" begin

let
    ex = CuError(CUDAdrv.SUCCESS)
    @test CUDAdrv.name(ex) == "SUCCESS"
    @test CUDAdrv.description(ex) == "no error"
    @test eval(Meta.parse(repr(ex))) == ex

    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test occursin("0", str)
    @test occursin("no error", str)
end

let
    ex = CuError(CUDAdrv.SUCCESS, "foobar")

    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test occursin("foobar", str)
end

end
