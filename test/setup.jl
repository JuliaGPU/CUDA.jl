using Test
using CUDA
using CUDACore
using GPUArrays
using NVML: has_nvml, NVML
using ParallelTestRunner
using ParallelTestRunner: AbstractTestRecord, TestRecord, WorkerTestSet
using Test: DefaultTestSet
using Printf: @sprintf
using Random

# ensure CUDA.jl is functional
@assert CUDA.functional(true)

# GPUArrays has a testsuite that isn't part of the main package; include it
# directly. After this runs, module `TestSuite` is available to tests.
let gpuarrays = pathof(GPUArrays),
    gpuarrays_root = dirname(dirname(gpuarrays))
    include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
end

if VERSION >= v"1.13.0-DEV.1044"
    using Base.ScopedValues
end

# detect compute-sanitizer, to disable incompatible tests (e.g. using CUPTI)
const sanitize = any(contains("NV_SANITIZER"), keys(ENV))

# in addition, CUPTI is not available on older GPUs with recent CUDA toolkits
function can_use_cupti()
    sanitize && return false

    # NVIDIA bug #3964667: CUPTI in CUDA 11.7+ broken for sm_35 devices
    if CUDA.runtime_version() >= v"11.7" && capability(device()) <= v"3.7"
        return false
    end

    # Tegra requires running as root and modifying the device tree
    if CUDA.is_tegra()
        return false
    end

    return true
end

# precompile the runtime library
CUDA.precompile_runtime()


## custom test record capturing CUDA-specific statistics

struct CUDATestRecord <: AbstractTestRecord
    base::TestRecord
    gpu_bytes::UInt64
    gpu_time::Float64
    gpu_rss::Union{UInt64, Missing}
end

# GPU per-process memory via NVML. Returns `missing` in containers or on
# devices without NVML support.
function gpu_rss_nvml()
    has_nvml() || return missing
    cuda_dev = device()
    mig = uuid(cuda_dev) != parent_uuid(cuda_dev)
    nvml_dev = NVML.Device(uuid(cuda_dev); mig)
    try
        gpu_processes = NVML.compute_processes(nvml_dev)
        if haskey(gpu_processes, getpid())
            return gpu_processes[getpid()].used_gpu_memory
        else
            return missing
        end
    catch err
        (isa(err, NVML.NVMLError) && err.code == NVML.ERROR_NOT_SUPPORTED) || rethrow()
        return missing
    end
end

function ParallelTestRunner.execute(::Type{CUDATestRecord}, mod::Module, f, name,
                                    start_time, custom_args)
    # Context-destroying tests use plain Julia timing: CUDA events can't survive
    # the context teardown. Delegate to PTR's default `execute` for those and
    # wrap with empty GPU stats.
    if name in custom_args.julia_timed_tests
        base = ParallelTestRunner.execute(TestRecord, mod, f, name, start_time, custom_args)
        return CUDATestRecord(base, UInt64(0), 0.0, missing)
    end

    data = @eval mod begin
        GC.gc(true)
        Random.seed!(1)
        stats = CUDA.@timed @testset WorkerTestSet "placeholder" begin
            @testset DefaultTestSet $name begin
                $f
            end
        end
        (; testset = stats.value,
           stats.time,
           cpu_bytes = UInt64(stats.cpu_bytes),
           cpu_gctime = Float64(stats.cpu_gctime),
           gpu_bytes = UInt64(stats.gpu_bytes),
           gpu_time = Float64(stats.gpu_memtime))
    end

    rss = Sys.maxrss()
    base = TestRecord(data.testset, data.time, data.cpu_bytes, data.cpu_gctime,
                      0.0, rss, time() - start_time)
    record = CUDATestRecord(base, data.gpu_bytes, data.gpu_time, gpu_rss_nvml())
    GC.gc(true)
    CUDA.reclaim()
    return record
end


## print overrides: extend the default layout with GPU columns

const GPU_TIME_ALIGN = textwidth("GC (s)")
const GPU_ALLOC_ALIGN = textwidth("Alloc (MB)")
const GPU_RSS_ALIGN = textwidth("RSS (MB)")

function ParallelTestRunner.print_header(::Type{CUDATestRecord}, ctx::ParallelTestRunner.TestIOContext,
                                         testgroupheader, workerheader)
    lock(ctx.lock)
    try
        # upper band
        printstyled(ctx.stdout, " "^(ctx.name_align + textwidth(testgroupheader) - 3), " │ ", color = :white)
        printstyled(ctx.stdout, "  Test   │", color = :white)
        ctx.verbose && printstyled(ctx.stdout, "   Init   │", color = :white)
        VERSION >= v"1.11" && ctx.verbose && printstyled(ctx.stdout, " Compile │", color = :white)
        printstyled(ctx.stdout, " ──────────── GPU ───────────── │", color = :white)
        printstyled(ctx.stdout, " ──────────────── CPU ──────────────── │\n", color = :white)

        # lower band
        printstyled(ctx.stdout, testgroupheader, color = :white)
        printstyled(ctx.stdout, lpad(workerheader, ctx.name_align - textwidth(testgroupheader) + 1), " │ ", color = :white)
        printstyled(ctx.stdout, "time (s) │", color = :white)
        ctx.verbose && printstyled(ctx.stdout, " time (s) │", color = :white)
        VERSION >= v"1.11" && ctx.verbose && printstyled(ctx.stdout, "   (%)   │", color = :white)
        printstyled(ctx.stdout, " GC (s) │ Alloc (MB) │ RSS (MB) │", color = :white)
        printstyled(ctx.stdout, " GC (s) │ GC % │ Alloc (MB) │ RSS (MB) │\n", color = :white)
        flush(ctx.stdout)
    finally
        unlock(ctx.lock)
    end
end

function print_cuda_row(io::IO, record::CUDATestRecord, wrkr, test, ctx::ParallelTestRunner.TestIOContext;
                       color::Symbol = :white)
    base = record.base
    printstyled(io, test, color = color)
    printstyled(io, lpad("($wrkr)", ctx.name_align - textwidth(test) + 1, " "), " │ ", color = color)

    time_str = @sprintf("%7.2f", base.time)
    printstyled(io, lpad(time_str, ctx.elapsed_align, " "), " │ ", color = color)

    if ctx.verbose
        init_time_str = @sprintf("%7.2f", base.total_time - base.time)
        printstyled(io, lpad(init_time_str, ctx.elapsed_align, " "), " │ ", color = color)
        if VERSION >= v"1.11"
            ct = base.time > 0 ? 100 * base.compile_time / base.time : 0.0
            ct_str = @sprintf("%7.2f", Float64(ct))
            printstyled(io, lpad(ct_str, ctx.compile_align, " "), " │ ", color = color)
        end
    end

    # GPU columns
    gpu_time_str = @sprintf("%5.2f", record.gpu_time)
    printstyled(io, lpad(gpu_time_str, GPU_TIME_ALIGN, " "), " │ ", color = color)
    gpu_alloc_str = @sprintf("%5.2f", record.gpu_bytes / 2^20)
    printstyled(io, lpad(gpu_alloc_str, GPU_ALLOC_ALIGN, " "), " │ ", color = color)
    gpu_rss_str = ismissing(record.gpu_rss) ? "N/A" : @sprintf("%5.2f", record.gpu_rss / 2^20)
    printstyled(io, lpad(gpu_rss_str, GPU_RSS_ALIGN, " "), " │ ", color = color)

    # CPU columns
    gc_str = @sprintf("%5.2f", base.gctime)
    printstyled(io, lpad(gc_str, ctx.gc_align, " "), " │ ", color = color)
    pct = base.time > 0 ? 100 * base.gctime / base.time : 0.0
    pct_str = @sprintf("%4.1f", pct)
    printstyled(io, lpad(pct_str, ctx.percent_align, " "), " │ ", color = color)
    alloc_str = @sprintf("%5.2f", base.bytes / 2^20)
    printstyled(io, lpad(alloc_str, ctx.alloc_align, " "), " │ ", color = color)
    rss_str = @sprintf("%5.2f", base.rss / 2^20)
    printstyled(io, lpad(rss_str, ctx.rss_align, " "), " │\n", color = color)
end

function ParallelTestRunner.print_test_finished(record::CUDATestRecord, wrkr, test,
                                                ctx::ParallelTestRunner.TestIOContext)
    lock(ctx.lock)
    try
        print_cuda_row(ctx.stdout, record, wrkr, test, ctx; color = :white)
        flush(ctx.stdout)
    finally
        unlock(ctx.lock)
    end
end

function ParallelTestRunner.print_test_failed(record::CUDATestRecord, wrkr, test,
                                              ctx::ParallelTestRunner.TestIOContext)
    lock(ctx.lock)
    try
        print_cuda_row(ctx.stderr, record, wrkr, test, ctx; color = :red)
        flush(ctx.stderr)
    finally
        unlock(ctx.lock)
    end
end


nothing # File is loaded via include; ensure it returns "nothing".
