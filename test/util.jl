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
                ret, read(fname, String)
            finally
                rm(fname, force=true)
            end
        end
    end
end

# variant on @test_throws that checks the CuError error code
macro test_throws_cuerror(code, ex)
    # generate a test only returning CuError if it is the correct one
    test = quote
        try
            $(esc(ex))
        catch err
            isa(err, CuError) || rethrow()
            err.code == $code || error("Wrong CuError: ", err.code, " instead of ", $code)
            rethrow()
        end
    end

    # now re-use @test_throws (which ties into @testset, etc)
    quote
        @test_throws CuError $test
    end
end

mutable struct NoThrowTestSet <: Test.AbstractTestSet
    results::Vector
    NoThrowTestSet(desc) = new([])
end
Test.record(ts::NoThrowTestSet, t::Test.Result) = (push!(ts.results, t); t)
Test.finish(ts::NoThrowTestSet) = ts.results
fails = @testset NoThrowTestSet begin
    # OK
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CuError(CUDAdrv.ERROR_UNKNOWN))
    # Fail, wrong CuError
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CuError(CUDAdrv.ERROR_INVALID_VALUE))
    # Fail, wrong Exception
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN error()
end
@test isa(fails[1], Test.Pass)
@test isa(fails[2], Test.Fail)
@test isa(fails[3], Test.Fail)

function julia_cmd(cmd)
    return `
        $(Base.julia_cmd())
        --color=$(Base.have_color ? "yes" : "no")
        --compiled-modules=$(Base.JLOptions().use_compiled_modules != 0 ? "yes" : "no")
        --history-file=no
        --startup-file=$(Base.JLOptions().startupfile != 2 ? "yes" : "no")
        --code-coverage=$(["none", "user", "all"][1+Base.JLOptions().code_coverage])
        $cmd
    `
end

