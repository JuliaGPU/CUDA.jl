using Test
using CUDA
using CUDACore
using FileCheck
using GPUArrays
using NVML: has_nvml, NVML
using ParallelTestRunner
using ParallelTestRunner: AbstractTestRecord, TestRecord, WorkerTestSet
using Test: DefaultTestSet
using Printf: @sprintf
using Random
using StyledStrings: @styled_str

# ensure CUDA.jl is functional
@assert CUDA.functional(true)

# GPUArrays has a testsuite that isn't part of the main package; include it
# directly. After this runs, module `TestSuite` is available to tests.
let gpuarrays = pathof(GPUArrays),
    gpuarrays_root = dirname(dirname(gpuarrays))
    include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
end
TestSuite.sparse_types(::Type{<:CuArray}) =
    (CUDA.CuSparseVector, CUDA.CuSparseMatrixCSC, CUDA.CuSparseMatrixCSR)

function TestSuite.supported_eltypes(::Type{<:CuArray}, test)
    typs = [TestSuite.supported_eltypes(CuArray)...]
    if startswith(string(test), "test_sparse")
        filter!(ET -> !(ET <: Integer || ET <: Complex{<:Integer}), typs)
    end
    return typs
end

if VERSION >= v"1.13.0-DEV.1044"
    using Base.ScopedValues
end

# detect compute-sanitizer, to disable incompatible tests (e.g. using CUPTI)
const sanitize = any(contains("NV_SANITIZER"), keys(ENV))

# in addition, CUPTI is not available on older GPUs with recent CUDA toolkits
function can_use_cupti()
    sanitize && return false

    # Tegra requires running as root and modifying the device tree
    if CUDA.is_tegra()
        return false
    end

    return true
end

# precompile the runtime library
CUDA.precompile_runtime()

# Cap the amount of memory the CUDA pool keeps cached after frees (default:
# unbounded). Above this watermark, freed stream-ordered buffers go back to
# the driver, which keeps each test's GPU RSS close to its peak live
# allocation rather than the running max across the test's lifetime. The
# threshold trades per-alloc pool-refill cost for lower NVML-reported memory;
# tune up if test wall-time regresses, down if a worker's RSS budget is tight.
const pool_release_threshold = 256 * 2^20  # 256 MiB
let dev = device()
    if CUDACore.stream_ordered(dev)
        pool = CUDACore.pool_create(dev)
        CUDACore.attribute!(pool, CUDACore.MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            UInt64(pool_release_threshold))
    end
end


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
        name_pad_str = " "^(ctx.name_align + textwidth(testgroupheader) - 3) * " │ "
        init_str = ctx.verbose ? "   Init   │" : ""
        compile_str = VERSION >= v"1.11" && ctx.verbose ? " Compile │" : ""
        header_top_str = styled"{ptr_default:$name_pad_str  Test   │$init_str$compile_str ──────────── GPU ───────────── │ ──────────────── CPU ──────────────── │}\n"
        print(ctx.stdout, header_top_str)

        # lower band
        workerheaderstr = lpad(workerheader, ctx.name_align - textwidth(testgroupheader) + 1)
        init_time_str = ctx.verbose ? " time (s) │" : ""
        comp_time_str = VERSION >= v"1.11" && ctx.verbose ? "   (%)   │" : ""
        header_bottom_str = styled"{ptr_default:$testgroupheader$workerheaderstr │ time (s) │$init_time_str$comp_time_str GC (s) │ Alloc (MB) │ RSS (MB) │ GC (s) │ GC % │ Alloc (MB) │ RSS (MB) │}\n"
        print(ctx.stdout, header_bottom_str)
        flush(ctx.stdout)
    finally
        unlock(ctx.lock)
    end
end

function print_cuda_row(io::IO, record::CUDATestRecord, wrkr, test, ctx::ParallelTestRunner.TestIOContext;
                       face::Symbol = :ptr_default)
    base = record.base
    padded_wrkr = lpad("($wrkr)", ctx.name_align - textwidth(test) + 1, " ")

    time_str = @sprintf("%7.2f", base.time)
    padded_time = lpad(time_str, ctx.elapsed_align, " ")

    padded_init_time, padded_comp_time = if ctx.verbose
        init_time_str = @sprintf("%7.2f", base.total_time - base.time)
        init_time = lpad(init_time_str, ctx.elapsed_align, " ") * " │ "
        comp_time = if VERSION >= v"1.11"
            ct = base.time > 0 ? 100 * base.compile_time / base.time : 0.0
            ct_str = @sprintf("%7.2f", Float64(ct))
            lpad(ct_str, ctx.compile_align, " ") * " │ "
        else
            ""
        end
        init_time, comp_time
    else
        "", ""
    end

    # GPU columns
    gpu_time_str = @sprintf("%5.2f", record.gpu_time)
    padded_gpu_time = lpad(gpu_time_str, GPU_TIME_ALIGN, " ")
    gpu_alloc_str = @sprintf("%5.2f", record.gpu_bytes / 2^20)
    padded_gpu_alloc = lpad(gpu_alloc_str, GPU_ALLOC_ALIGN, " ")
    gpu_rss_str = ismissing(record.gpu_rss) ? "N/A" : @sprintf("%5.2f", record.gpu_rss / 2^20)
    padded_gpu_rss = lpad(gpu_rss_str, GPU_RSS_ALIGN, " ")

    # CPU columns
    gc_str = @sprintf("%5.2f", base.gctime)
    padded_gc = lpad(gc_str, ctx.gc_align, " ")
    pct = base.time > 0 ? 100 * base.gctime / base.time : 0.0
    padded_percent = lpad(@sprintf("%4.1f", pct), ctx.percent_align, " ")
    padded_alloc = lpad(@sprintf("%5.2f", base.bytes / 2^20), ctx.alloc_align, " ")
    padded_rss = lpad(@sprintf("%5.2f", base.rss / 2^20), ctx.rss_align, " ")

    # yellow when worker to be killed unless it's a fail
    mem_face = mem_use > ctx.max_worker_rss ? (face == :ptr_error ? :ptr_error : :ptr_warn) : :ptr_default
    out_str = styled"{$face:$test$padded_wrkr │ $padded_time │ $padded_init_time$padded_comp_time$padded_gpu_time │ $padded_gpu_alloc │ $padded_gpu_rss │ $padded_gc │ $padded_percent │ $padded_alloc │ {$mem_face:$padded_rss} │}\n"
    print(io, out_str)
end

function ParallelTestRunner.print_test_finished(record::CUDATestRecord, wrkr, test,
                                                ctx::ParallelTestRunner.TestIOContext)
    lock(ctx.lock)
    try
        print_cuda_row(ctx.stdout, record, wrkr, test, ctx; face = :ptr_default)
        flush(ctx.stdout)
    finally
        unlock(ctx.lock)
    end
end

function ParallelTestRunner.print_test_failed(record::CUDATestRecord, wrkr, test,
                                              ctx::ParallelTestRunner.TestIOContext)
    lock(ctx.lock)
    try
        print_cuda_row(ctx.stderr, record, wrkr, test, ctx; face = ctx.nonpass_face[])
        flush(ctx.stderr)
    finally
        unlock(ctx.lock)
    end
end


nothing # File is loaded via include; ensure it returns "nothing".
