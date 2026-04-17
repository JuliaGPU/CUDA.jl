using CUDA
using CUDACore
using cuBLAS, cuSPARSE, cuSOLVER, cuFFT, cuRAND
using cuDNN, cuTENSOR, cuTensorNet, cuStateVec
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
toolkit_release = Base.thisminor(CUDA.runtime_version())
label_match = match(r"^CUDA ([\d.]+)$", get(ENV, "BUILDKITE_LABEL", ""))
if label_match !== nothing
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

# subpackage tests under lib/*/test/ — include when requested via `--all` or
# by explicitly selecting a `libraries/*` positional.
const subpackages = ["cublas", "cusparse", "cusolver", "cufft", "curand",
                     "cudnn", "cutensor", "cutensornet", "custatevec"]
function include_subpackages!(testsuite)
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
    return testsuite
end

want_all = args.custom["all"] !== nothing
want_libraries = any(startswith("libraries/"), args.positionals)
if want_all || want_libraries
    include_subpackages!(testsuite)
end

# default filter (no positionals given): package extensions require extra
# deps not in test/Project.toml, and subpackages are opt-in via `--all`.
if isempty(args.positionals)
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

# pick the first visible device and use its free memory to cap worker count.
# (Set `CUDA_VISIBLE_DEVICES` to choose which device is used.)
first_gpu = first(devices())
gpu_free = device!(first_gpu) do
    mem = CUDA.free_memory()
    device_reset!()
    mem
end
gpu_jobs = max(1, Int(gpu_free) ÷ (2 * 2^30))

if args.jobs === nothing
    default_jobs = min(ParallelTestRunner.default_njobs(), gpu_jobs)
    args = ParallelTestRunner.ParsedArgs(
        Some(default_jobs), args.verbose, args.quickfail, args.list,
        args.custom, args.positionals,
    )
end


## compute-sanitizer wrapping (optional)

exename = nothing
if args.custom["sanitize"] !== nothing
    tool_val = args.custom["sanitize"].value
    tool = tool_val === nothing ? "memcheck" : tool_val
    # install CUDA_SDK_jll in a temporary environment
    project = Base.active_project()
    Pkg.activate(; temp = true)
    Pkg.add("CUDA_SDK_jll")
    @eval using CUDA_SDK_jll
    Pkg.activate(project)

    compute_sanitizer = joinpath(CUDA_SDK_jll.artifact_dir,
                                 "cuda", "compute-sanitizer", "compute-sanitizer")
    if Sys.iswindows()
        compute_sanitizer *= ".exe"
    end
    @info "Running under " * read(`$compute_sanitizer --version`, String)
    exename = `$compute_sanitizer --tool=$tool --launch-timeout=0 --target-processes=all --report-api-errors=no $(Base.julia_cmd()[1])`
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

runtests(CUDA, args;
         testsuite, init_code, init_worker_code, test_worker,
         RecordType = CUDATestRecord,
         custom_args = (; julia_timed_tests = context_destroying_tests),
         exename)
