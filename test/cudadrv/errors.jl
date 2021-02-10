@testset "errors" begin

@testcase "essentials" begin
    # make sure we're initialized, which influences error reporting
    context()

    ex = CuError(CUDA.SUCCESS)
    @test CUDA.name(ex) == "SUCCESS"
    @test CUDA.description(ex) == "no error"
    @test eval(Meta.parse(repr(ex))) == ex

    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test occursin("0", str)
    @test occursin("no error", str)
end

@testcase "repr" begin
    # make sure we're initialized, which influences error reporting
    context()

    ex = CuError(CUDA.SUCCESS, "foobar")

    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test occursin("foobar", str)
end

end
