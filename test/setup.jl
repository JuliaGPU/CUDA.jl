using Distributed, Test, CUDA
using CUDA: i32

# ensure CUDA.jl is functional
@assert CUDA.functional(true)

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

using Random

# detect compute-sanitizer, to disable incompatible tests (e.g. using CUPTI)
const sanitize = any(contains("NV_SANITIZER"), keys(ENV))

# in addition, CUPTI is not available on older GPUs with recent CUDA toolkits
function can_use_cupti()
    sanitize && return false

    # NVIDIA bug #3964667: CUPTI in CUDA 11.7+ broken for sm_35 devices
    if CUDA.runtime_version() >= v"11.7" && capability(device()) <= v"3.7"
        return false
    end

    # Tegra requires running as root and and modifying the device tree
    if CUDA.is_tegra()
        return false
    end

    return true
end

# precompile the runtime library
CUDA.precompile_runtime()


## entry point

function runtests(f, name, time_source=:cuda)
    old_print_setting = Test.TESTSET_PRINT_ENABLE[]
    Test.TESTSET_PRINT_ENABLE[] = false

    try
        # generate a temporary module to execute the tests in
        mod_name = Symbol("Test", rand(1:100), "Main_", replace(name, '/' => '_'))
        mod = @eval(Main, module $mod_name end)
        @eval(mod, using Test, Random, CUDA)

        let id = myid()
            wait(@spawnat 1 print_testworker_started(name, id))
        end

        ex = quote
            GC.gc(true)
            Random.seed!(1)

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
        cuda_dev = device()
        gpu_rss = if has_nvml()
            mig = uuid(cuda_dev) != parent_uuid(cuda_dev)
            nvml_dev = NVML.Device(uuid(cuda_dev); mig)
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
        if VERSION >= v"1.11.0-DEV.1529"
            tc = Test.get_test_counts(data[1])
            passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
                tc.passes, tc.fails, tc.errors, tc.broken, tc.cumulative_passes,
                tc.cumulative_fails, tc.cumulative_errors, tc.cumulative_broken
        else
            passes,fails,errors,broken,c_passes,c_fails,c_errors,c_broken =
                Test.get_test_counts(data[1])
        end
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

        GC.gc(true)
        res
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

                    # NOTE: CUDA requires a 'proper' sync to flush its printf buffer
                    synchronize(context())
                end
            end
            ret, read(fname, String)
        end
    end
end

# Run some code on-device
macro on_device(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    @gensym kernel
    esc(quote
        let
            function $kernel()
                $code
                return
            end

            CUDA.@sync @cuda $(kwargs...) $kernel()
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

function julia_exec(args::Cmd, env...)
    # FIXME: this doesn't work when the compute mode is set to exclusive
    cmd = Base.julia_cmd()
    cmd = `$cmd --project=$(Base.active_project()) --color=no $args`

    out = Pipe()
    err = Pipe()
    proc = run(pipeline(addenv(cmd, env...), stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)
    wait(proc)
    proc, read(out, String), read(err, String)
end

nothing # File is loaded via a remotecall to "include". Ensure it returns "nothing".
