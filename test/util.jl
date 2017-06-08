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
        catch err
            isa(err, CuError) || rethrow()
            err == $kind      || rethrow(ErrorException(string("Wrong CuError kind: ", err, " instead of ", $kind)))
            rethrow()
        end
    end

    # now re-use @test_throws (which ties into @testset, etc)
    quote
        @test_throws CuError $test
    end
end

type NoThrowTestSet <: Base.Test.AbstractTestSet
    results::Vector
    NoThrowTestSet(desc) = new([])
end
Base.Test.record(ts::NoThrowTestSet, t::Base.Test.Result) = (push!(ts.results, t); t)
Base.Test.finish(ts::NoThrowTestSet) = ts.results
fails = @testset NoThrowTestSet begin
    # OK
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CUDAdrv.ERROR_UNKNOWN)
    # Fail, wrong CuError
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CUDAdrv.ERROR_INVALID_VALUE)
    # Fail, wrong Exception
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN error()
end
@test isa(fails[1], Base.Test.Pass)
@test isa(fails[2], Base.Test.Fail)
@test isa(fails[3], Base.Test.Fail)

function julia_cmd(cmd)
    return `
        $(Base.julia_cmd())
        --color=$(Base.have_color ? "yes" : "no")
        --compilecache=$(Bool(Base.JLOptions().use_compilecache) ? "yes" : "no")
        --history-file=no
        --startup-file=$(Base.JLOptions().startupfile != 2 ? "yes" : "no")
        --code-coverage=$(["none", "user", "all"][1+Base.JLOptions().code_coverage])
        $cmd
    `
end
