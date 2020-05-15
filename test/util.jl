# @test_throw, with additional testing for the exception message
macro test_throws_message(f, typ, ex...)
    quote
        msg = ""
        @test_throws $(esc(typ)) try
            $(esc(ex...))
        catch err
            msg = sprint(showerror, err)
            rethrow()
        end

        if !$(esc(f))(msg)
            # @test should return its result, but doesn't
            @error "Failed to validate error message\n$msg"
        end
        @test $(esc(f))(msg)
    end
end

# @test_throw, peeking into the load error for testing macro errors
macro test_throws_macro(ty, ex)
    return quote
        Test.@test_throws $(esc(ty)) try
            $(esc(ex))
        catch err
            @test err isa LoadError
            @test err.file === $(string(__source__.file))
            @test err.line === $(__source__.line + 1)
            rethrow(err.error)
        end
    end
end

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))
                end
            end
            ret, read(fname, String)
        end
    end
end

# Run some code on-device, returning captured standard output
macro on_device(ex)
    @gensym kernel
    esc(quote
        let
            function $kernel()
                $ex
                return
            end

            @cuda $kernel()
            synchronize()
        end
    end)
end

# helper function for sinking a value to prevent the callee from getting optimized away
@inline sink(i::Int32) =
    Base.llvmcall("""%slot = alloca i32
                     store volatile i32 %0, i32* %slot
                     %value = load volatile i32, i32* %slot
                     ret i32 %value""", Int32, Tuple{Int32}, i)
@inline sink(i::Int64) =
    Base.llvmcall("""%slot = alloca i64
                     store volatile i64 %0, i64* %slot
                     %value = load volatile i64, i64* %slot
                     ret i64 %value""", Int64, Tuple{Int64}, i)

function julia_script(code, args=``)
    # FIXME: this doesn't work when the compute mode is set to exclusive
    script = """using CUDAnative, CUDAdrv
                const CuArray = CUDAnative.CuHostArray
                device!($(device()))

                $code"""
    cmd = Base.julia_cmd()
    if Base.JLOptions().project != C_NULL
        cmd = `$cmd --project=$(unsafe_string(Base.JLOptions().project))`
    end
    cmd = `$cmd --eval $script $args`

    out = Pipe()
    err = Pipe()
    proc = run(pipeline(cmd, stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)
    wait(proc)
    proc.exitcode, read(out, String), read(err, String)
end

# tests that are conditionall broken
macro test_broken_if(cond, ex...)
    quote
        if $(esc(cond))
            @test_broken $(map(esc, ex)...)
        else
            @test $(map(esc, ex)...)
        end
    end
end
