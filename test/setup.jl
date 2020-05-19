using Test, CUDA

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using Random


## entry point

function runtests(name, device=nothing)
    old_print_setting = Test.TESTSET_PRINT_ENABLE[]
    Test.TESTSET_PRINT_ENABLE[] = false
    try
        ex = quote
            Random.seed!(1)
            CUDA.allowscalar(false)
            if $device !== nothing
                device!($device)
                CUDA.@timed @testset $"$name" begin
                    include($"$(@__DIR__)/$name.jl")
                end
            else
                # take care not to initialize the device
                res = @timed @testset $"$name" begin
                    include($"$(@__DIR__)/$name.jl")
                end
                res..., 0, 0, 0
            end
        end
        res_and_time_data = Core.eval(Main, ex)
        rss = Sys.maxrss()
        # TODO: GPU RSS using nvmlDeviceGetComputeRunningProcesses
        #res_and_time_data[1] is the testset
        passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
            Test.get_test_counts(res_and_time_data[1])
        if res_and_time_data[1].anynonpass == false
            res_and_time_data = (
                                 (passes+c_passes,broken+c_broken),
                                 res_and_time_data[2],
                                 res_and_time_data[3],
                                 res_and_time_data[4],
                                 res_and_time_data[5],
                                 res_and_time_data[6],
                                 res_and_time_data[7],
                                 res_and_time_data[8])
        end
        vcat(collect(res_and_time_data), rss)
    finally
        Test.TESTSET_PRINT_ENABLE[] = old_print_setting
    end
end


## auxiliary stuff

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
    script = """using CUDA
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

# some tests are mysteriously broken with certain hardware/software.
# use a horrible macro to mark those tests as "potentially broken"
@eval Test begin
    export @test_maybe_broken

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

nothing # File is loaded via a remotecall to "include". Ensure it returns "nothing".
