using Distributed, Test, CUDA

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

using Random

# detect compute-sanitizer, to disable incompatible tests (e.g. using CUPTI),
# and to skip tests that are known to generate innocuous API errors
const sanitize = any(contains("NV_SANITIZER"), keys(ENV))
macro not_if_sanitize(ex)
    sanitize || return esc(ex)
    quote
        @test_skip $ex
    end
end

# precompile the runtime library
CUDA.precompile_runtime()

# for when we include tests directly
CUDA.allowscalar(false)


## entry point

function runtests(f, name, time_source=:cuda, snoop=nothing)
    old_print_setting = Test.TESTSET_PRINT_ENABLE[]
    Test.TESTSET_PRINT_ENABLE[] = false

    if snoop !== nothing
        io = open(snoop, "w")
        ccall(:jl_dump_compiles, Nothing, (Ptr{Nothing},), io.handle)
    end

    try
        # generate a temporary module to execute the tests in
        mod_name = Symbol("Test", rand(1:100), "Main_", replace(name, '/' => '_'))
        mod = @eval(Main, module $mod_name end)
        @eval(mod, using Test, Random, CUDA)

        let id = myid()
            wait(@spawnat 1 print_testworker_started(name, id))
        end

        ex = quote
            Random.seed!(1)
            CUDA.allowscalar(false)

            if $(QuoteNode(time_source)) == :cuda
                CUDA.@timed @testset $name begin
                    $f()
                end
            elseif $(QuoteNode(time_source)) == :julia
                res = @timed @testset $name begin
                    $f()
                end
                res..., 0, 0, 0
            else
                error("Unknown time source: " * $(string(time_source)))
            end
        end
        data = Core.eval(mod, ex)
        #data[1] is the testset

        # process results
        cpu_rss = Sys.maxrss()
        gpu_rss = if has_nvml()
            cuda_dev = device()
            nvml_dev = NVML.Device(uuid(cuda_dev))
            try
                gpu_processes = NVML.compute_processes(nvml_dev)
                if haskey(gpu_processes, getpid())
                    gpu_processes[getpid()].used_gpu_memory
                else
                    # happens when we didn't do compute, or when using containers:
                    # https://github.com/NVIDIA/gpu-monitoring-tools/issues/63
                    missing
                end
            catch err
                (isa(err, NVML.NVMLError) && err.code == NVML.ERROR_NOT_SUPPORTED) || rethrow()
                missing
            end
        else
            missing
        end
        passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
            Test.get_test_counts(data[1])
        if data[1].anynonpass == false
            data = ((passes+c_passes,broken+c_broken),
                    data[2],
                    data[3],
                    data[4],
                    data[5],
                    data[6],
                    data[7],
                    data[8])
        end
        res = vcat(collect(data), cpu_rss, gpu_rss)

        CUDA.can_reset_device() && device_reset!()
        res
    finally
        if snoop !== nothing
            ccall(:jl_dump_compiles, Nothing, (Ptr{Nothing},), C_NULL)
            close(io)
        end

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

                    # NOTE: CUDA requires a 'proper' sync to flush its printf buffer
                    device_synchronize()
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
    @gensym msg
    quote
        $msg = ""
        @test_throws $(esc(typ)) try
            $(esc(ex...))
        catch err
            $msg = sprint(showerror, err)
            rethrow()
        end

        if !$(esc(f))($msg)
            # @test should return its result, but doesn't
            @error "Failed to validate error message\n" * $msg
        end
        @test $(esc(f))($msg)
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
