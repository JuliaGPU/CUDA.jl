using CUDAdrv
using Base.Test

using Compat

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        let fname = tempname()
            try
                ret = nothing
                open(fname, "w") do fout
                    redirect_stdout(fout) do
                        ret = $(esc(ex))
                    end
                end
                ret, readstring(fname)
            finally
                rm(fname, force=true)
            end
        end
    end
end

# variant on @test_throws that checks the CuError error code
macro test_throws_cuerror(kind, ex)
    # generate a test only returning CuError if it is the correct one
    test = quote
        try
            $(esc(ex))
        catch ex
            isa(ex, CuError) || rethrow()
            ex == $kind      || rethrow(ErrorException(string("Wrong CuError kind: ", ex, " instead of ", $kind)))
            rethrow()
        end
    end

    # now re-use @test_throws (which ties into @testset, etc)
    quote
        @test_throws CuError $test
    end
end
@test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CUDAdrv.ERROR_UNKNOWN)
@test_throws Exception begin
    OLD_STDOUT = STDOUT
    redirect_stdout()
    try
        @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CUDAdrv.ERROR_INVALID_VALUE)
    finally
        redirect_stdout(OLD_STDOUT)
    end
end
@test_throws Exception begin
    OLD_STDOUT = STDOUT
    redirect_stdout()
    try
        @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN error()
    finally
        redirect_stdout(OLD_STDOUT)
    end
end

@testset "CUDAdrv" begin

@test devcount() > 0
if devcount() > 0
    include("core.jl")

    # pick most recent device (based on compute capability)
    global dev = nothing
    for i in 0:devcount()-1
        newdev = CuDevice(i)
        if dev == nothing || capability(newdev) > capability(dev)
            dev = newdev
        end
    end
    info("Testing using device $(name(dev))")
    global ctx = CuContext(dev, CUDAdrv.SCHED_BLOCKING_SYNC)

    include("wrappers.jl")
    include("gc.jl")

    include("examples.jl")
end
end