const ci_cache = GPUCompiler.CodeCache()

function CUDACompilerTarget(dev::CuDevice; kwargs...)
    cap = supported_capability(dev)
    ptx = v"6.3"    # we only need 6.2, but NVPTX doesn't support that

    exitable = true
    if cap < v"7"
        # JuliaGPU/CUDAnative.jl#4
        # ptxas for old compute capabilities has a bug where it messes up the
        # synchronization stack in the presence of shared memory and thread-divergent exit.
        exitable = false
    end
    if !has_nvml() || NVML.driver_version() < v"460"
        # JuliaGPU/CUDA.jl#431
        # TODO: tighten this conditional
        exitable = false
    end

    debuginfo = false

    PTXCompilerTarget(; cap, ptx, exitable, debuginfo, kwargs...)
end

struct CUDACompilerParams <: AbstractCompilerParams end

CUDACompilerJob = CompilerJob{PTXCompilerTarget,CUDACompilerParams}

GPUCompiler.runtime_module(::CUDACompilerJob) = CUDA

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(job::CUDACompilerJob, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(fn)},
           job, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

function GPUCompiler.link_libraries!(job::CUDACompilerJob, mod::LLVM.Module,
                                     undefined_fns::Vector{String})
    invoke(GPUCompiler.link_libraries!,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(mod), typeof(undefined_fns)},
           job, mod, undefined_fns)
    link_libdevice!(mod, job.target.cap, undefined_fns)
end

GPUCompiler.ci_cache(::CUDACompilerJob) = ci_cache

GPUCompiler.method_table(::CUDACompilerJob) = method_table
