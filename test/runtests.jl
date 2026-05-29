@static if VERSION < v"1.11" && get(ENV, "BUILDKITE_PIPELINE_NAME", "CUDA.jl") == "CUDA.jl"
    using Pkg
    Pkg.add(url="https://github.com/christiangnrd/KernelAbstractions.jl", rev="subgroups")
end

using CUDA
using CUDACore
using cuBLAS, cuSPARSE, cuSOLVER, cuFFT, cuRAND
using cuDNN, cuTENSOR, cuTensorNet, cuStateVec
using BFloat16s
using GPUArrays
using ParallelTestRunner
using Pkg
using InteractiveUtils: versioninfo
using Base.Filesystem: path_separator

# load helpers, CUDATestRecord, and ParallelTestRunner overrides on the main
# process. The same file is re-`include`d on every worker via `init_worker_code`
# below; the record type must exist on both sides of the Malt boundary.
include(joinpath(@__DIR__, "setup.jl"))

# parse command-line arguments (--sanitize[=tool] and --all are CUDA-specific)
args = parse_args(ARGS; custom = ["sanitize", "all"])

# check that CI is using the requested toolkit
label_match = match(r"^CUDA ([\d.]+)$", get(ENV, "BUILDKITE_LABEL", ""))
if label_match !== nothing
    toolkit_release = Base.thisminor(CUDA.runtime_version())
    @test toolkit_release == VersionNumber(label_match.captures[1])
end

# forcibly precompile the current environment in parallel (Pkg somehow ignores
# the dependencies that are pointed through via [sources])
Pkg.precompile()

@info "Julia information:\n" * sprint(io -> versioninfo(io))
@info "CUDA information:\n" * sprint(io -> CUDA.versioninfo(io))


## test discovery

# core tests from the top-level test directory, plus GPUArrays' testsuite.
testsuite = find_tests(@__DIR__)
delete!(testsuite, "setup")

# scalar-indexing tests only make sense when device memory supports it
for name in keys(TestSuite.tests)
    testsuite["gpuarrays/$name"] = :(TestSuite.tests[$name](CuArray))
end
if CUDACore.default_memory != CUDA.DeviceMemory
    delete!(testsuite, "gpuarrays/indexing scalar")
end

# BFloat16 kernel tests target native bf16 codegen (JuliaGPU/CUDA.jl#2441);
# skip unless BFloat16s emits native bf16 LLVM IR, the NVPTX backend can lower
# it (LLVM 17+; 1.11/LLVM 16 miscompiles bf16), and the device speaks PTX bf16
# (sm_80+).
if !BFloat16s.llvm_arithmetic ||
   Base.libllvm_version < v"17" ||
   capability(device()) < v"8.0"
    delete!(testsuite, "core/device/bfloat16")
end

# subpackage tests under lib/*/test/
const subpackages = ["cublas", "cusparse", "cusolver", "cufft", "curand",
                     "cudnn", "cutensor", "cutensornet", "custatevec"]
for pkg in subpackages
    testdir = normpath(@__DIR__, "..", "lib", pkg, "test")
    isdir(testdir) || continue
    sub_tests = find_tests(testdir)
    delete!(sub_tests, "setup")
    delete!(sub_tests, "runtests")
    setuppath = joinpath(testdir, "setup.jl")
    projectpath = joinpath(testdir, "Project.toml")
    for (name, include_expr) in sub_tests
        # include_expr is of the form `:(include($path))`
        path = include_expr.args[end]
        testsuite["libraries/$pkg/$name"] = quote
            old_project = Base.active_project()
            try
                if isfile($projectpath)
                    Base.set_active_project($projectpath)
                end
                include($setuppath)
                include($path)
            finally
                Base.set_active_project(old_project)
            end
        end
    end
end

# default filter (no positionals given): package extensions require extra
# deps not in test/Project.toml, and subpackages are opt-in via `--all`.
if isempty(args.positionals)
    want_all = args.custom["all"] !== nothing
    filter!(testsuite) do (name, _)
        if startswith(name, "extensions")
            return false
        end
        if startswith(name, "libraries/") && !want_all
            return false
        end
        return true
    end
end


## GPU-memory-based parallelism

# Cap worker count by how much of the primary device's free memory each worker
# claims. A CUDA worker needs its own context + libraries (~0.5–1 GiB baseline)
# plus room for peak per-test allocations; with pool caching capped in
# `test/setup.jl`, 1 GiB is enough.
# (Set `CUDA_VISIBLE_DEVICES` to choose which device is used.)
const gpu_memory_per_worker = 1 * 2^30
first_gpu = first(devices())
# Query free memory without creating a CUDA context on this coordinator process.
# NVML reads it straight from the driver (no context), whereas `CUDA.free_memory()`
# would spin up this device's primary context (~0.5 GiB) that we'd then hold for the
# entire run. Fall back to the CUDA query only when NVML is unavailable (e.g. WSL).
gpu_free = if has_nvml()
    mig = uuid(first_gpu) != parent_uuid(first_gpu)
    Int(NVML.memory_info(NVML.Device(uuid(first_gpu); mig)).free)
else
    device!(first_gpu) do
        Int(CUDA.free_memory())
    end
end
gpu_jobs = max(1, gpu_free ÷ gpu_memory_per_worker)

@info "Parallelism budget" device = "$(CUDA.name(first_gpu)) ($(deviceid(first_gpu)))" gpu_free = Base.format_bytes(gpu_free) gpu_jobs cpu_threads = Sys.CPU_THREADS cpu_free = Base.format_bytes(Sys.free_memory())

if args.jobs === nothing
    default_jobs = min(ParallelTestRunner.default_njobs(), gpu_jobs)
    args = ParallelTestRunner.ParsedArgs(
        Some(default_jobs), args.verbose, args.quickfail, args.list,
        args.custom, args.positionals,
    )
end


## compute-sanitizer wrapping (optional)

exename = nothing
sanitizer_log_dir = nothing
if args.custom["sanitize"] !== nothing
    tool_val = args.custom["sanitize"].value
    tool = tool_val === nothing ? "memcheck" : tool_val
    # install CUDA_SDK_jll in a temporary environment so we only grab the
    # tool for this run.
    project = Base.active_project()
    Pkg.activate(; temp = true, io = devnull)
    Pkg.add("CUDA_SDK_jll"; io = devnull)
    @eval using CUDA_SDK_jll
    Pkg.activate(project, io = devnull)

    compute_sanitizer = joinpath(CUDA_SDK_jll.artifact_dir,
                                 "cuda", "compute-sanitizer", "compute-sanitizer")
    if Sys.iswindows()
        compute_sanitizer *= ".exe"
    end
    @info "Running under " * read(`$compute_sanitizer --version`, String)
    # Route sanitizer output to per-process log files: its banner/reports on
    # stdout would otherwise collide with Malt's port handshake on the worker.
    # `%p` expands to the sanitizer process's PID. We surface relevant logs
    # after the run finishes.
    sanitizer_log_dir = mktempdir(prefix = "cuda-sanitizer-")
    log_pattern = joinpath(sanitizer_log_dir, "%p.log")
    exename = `$compute_sanitizer --tool=$tool --launch-timeout=0 --target-processes=all --report-api-errors=no --log-file=$log_pattern $(Base.julia_cmd()[1])`
end


## worker setup

const init_worker_code = quote
    include($(joinpath(@__DIR__, "setup.jl")))
end

const init_code = quote
    include($(joinpath(@__DIR__, "helpers.jl")))
end

# `core/initialization` and `core/cudadrv` destroy the CUDA context mid-test.
# CUDA events become invalid once the context is gone, so these tests use plain
# Julia timing. We also isolate them on a fresh worker so subsequent tests
# aren't affected by the torn-down context.
const context_destroying_tests = ("core/initialization", "core/cudadrv")
function test_worker(name, init_worker_code)
    if name in context_destroying_tests
        return addworker(; exename, init_worker_code)
    end
    return nothing
end


## run

function report_sanitizer_logs(log_dir)
    log_dir === nothing && return
    files = filter(endswith(".log"), readdir(log_dir; join = true))
    isempty(files) && return
    total_workers = length(files)
    flagged = String[]
    for file in files
        content = read(file, String)
        if !occursin("ERROR SUMMARY: 0 errors", content)
            push!(flagged, file)
        end
    end
    println()
    if isempty(flagged)
        printstyled("compute-sanitizer: $total_workers workers checked, no errors\n";
                    color = :green, bold = true)
    else
        printstyled("compute-sanitizer: $(length(flagged))/$total_workers workers reported issues:\n";
                    color = :red, bold = true)
        for file in flagged
            println("\n--- ", basename(file), " ---")
            print(read(file, String))
        end
    end
    println()
end

try
    runtests(CUDA, args;
             testsuite, init_code, init_worker_code, test_worker,
             RecordType = CUDATestRecord,
             custom_args = (; julia_timed_tests = context_destroying_tests),
             exename)
finally
    report_sanitizer_logs(sanitizer_log_dir)
end
