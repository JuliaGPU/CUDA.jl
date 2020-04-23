## target

struct CUDACompilerTarget <: CompositeCompilerTarget
    parent::PTXCompilerTarget

    CUDACompilerTarget(cap::VersionNumber) = new(PTXCompilerTarget(cap))
end

Base.parent(target::CUDACompilerTarget) = target.parent

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(target::CUDACompilerTarget, fn::String) =
    GPUCompiler.isintrinsic(target.parent, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

GPUCompiler.runtime_module(::CUDACompilerTarget) = CUDAnative


## job

struct CUDACompilerJob <: CompositeCompilerJob
    parent::PTXCompilerJob
end

CUDACompilerJob(target::AbstractCompilerTarget, source::FunctionSpec; kwargs...) =
    CUDACompilerJob(PTXCompilerJob(target, source; kwargs...))

Base.similar(job::CUDACompilerJob, source::FunctionSpec; kwargs...) =
    CUDACompilerJob(similar(job.parent, source; kwargs...))

Base.parent(job::CUDACompilerJob) = job.parent

function GPUCompiler.process_module!(job::CUDACompilerJob, mod::LLVM.Module)
    GPUCompiler.process_module!(job.parent, mod)

    emit_exception_flag!(mod)
end

function GPUCompiler.link_libraries!(job::CUDACompilerJob, mod::LLVM.Module,
                                     undefined_fns::Vector{String})
    GPUCompiler.link_libraries!(job.parent, mod, undefined_fns)

    # FIXME: this is ugly
    link_libdevice!(mod, job.parent.target.parent.cap, undefined_fns)
end
