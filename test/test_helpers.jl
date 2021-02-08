module TestHelpers

import Test
using XUnit

using CUDA

export TestSuite, testf,
       @not_if_memcheck, @run_if, @grab_output,
       @test_throws_cuerror, @test_throws_message, @test_throws_macro,
       @test_broken_if, @test_maybe_broken,
       @on_device, @in_module,
       sink, julia_script

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

# detect cuda-memcheck, to disable testts that are known to fail under cuda-memcheck
# (e.g. those using CUPTI) or result in verbose output (deliberate API errors)
const memcheck = isdefined(Main, :do_memcheck) ? Main.do_memcheck : haskey(ENV, "CUDA_MEMCHECK")
macro not_if_memcheck(ex)
    memcheck || return esc(ex)
    quote
        @test_skip $ex
    end
end

# check if a test is supported, and skip it if not. error if we're not allowed to skip.
const thorough = isdefined(Main, :do_thorough) ? Main.do_thorough : true
macro run_if(check, test)
    if thorough
        quote
            supported = $(esc(check))
            if !supported
                error("Unsupported")
            end
            $(esc(test))
        end
    else
        quote
            supported = $(esc(check))
            if supported
                $(esc(test))
            end
        end
    end
end

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
# NOTE: we grab the iolock here, but that's a reentrant lock, so we should not
#       switch tasks (e.g. by yielding via synchronize)!
macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                Base.iolock_begin()
                # XXX: this doesn't seem sufficient to keep XUnit.jl from generating output?
                XUnit.TESTSET_PRINT_ENABLE[] = false
                redirect_stdout(fout) do
                    ret = $(esc(ex))

                    # NOTE: CUDA requires a 'proper' sync to flush its printf buffer
                    synchronize_all()
                end
                XUnit.TESTSET_PRINT_ENABLE[] = true
                Base.iolock_end()
            end
            ret, read(fname, String)
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

# Run some code on-device
macro on_device(ex)
    @gensym kernel
    esc(quote
        let
            function $kernel()
                $ex
                return
            end

            CUDA.@sync @cuda $kernel()
        end
    end)
end

# Put some code in a private module, and return a handle to it.
# This is useful for kernel definitions, which often contain boxes
# when defined in local scope. The disadvantage is that calls to
# functions in these modules need to be invokelatest'd
# (but not for kernel code because of JuliaGPU/GPUCompiler.jl#146).
macro in_module(ex)
    quote
        mod = eval(Expr(:module, true, gensym("PrivateModule"), quote end))
        Base.eval(mod, quote
            using ..CUDA, ..XUnit, ..TestHelpers
            import Test
        end)
        Base.eval(mod, $(esc(ex)))
        mod
    end
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
    script = """using CUDA
                device!($(device()))

                $code"""
    cmd = Base.julia_cmd()
    if Base.JLOptions().project != C_NULL
        cmd = `$cmd --project=$(unsafe_string(Base.JLOptions().project))`
    end
    cmd = `$cmd --color=no --eval $script $args`

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

# some tests are mysteriously broken with certain hardware/software.
# use a horrible macro to mark those tests as "potentially broken"
@eval Test begin
    macro test_maybe_broken(ex, kws...)
        test_expr!("@test_maybe_broken", ex, kws...)
        orig_ex = Expr(:inert, ex)
        result = get_test_result(ex, __source__)
        quote
            x = $result
            if x.value
                do_test(x, $orig_ex)
            else
                do_broken_test(x, $orig_ex)
            end
        end
    end
end
using Test: @test_maybe_broken

end


## global set-up

# here, and not in test_all.jl, to make it possible to
# include this file and run a single test suite.

using XUnit, CUDA, .TestHelpers
import Test

CUDA.allowscalar(false)
