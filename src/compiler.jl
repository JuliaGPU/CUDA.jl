struct CUDACompilerParams <: AbstractCompilerParams end

CUDACompilerJob = CompilerJob{PTXCompilerTarget,CUDACompilerParams}

GPUCompiler.runtime_module(::CUDACompilerJob) = CUDAnative

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(job::CUDACompilerJob, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(fn)},
           job, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

function GPUCompiler.process_module!(job::CUDACompilerJob, mod::LLVM.Module)
    invoke(GPUCompiler.process_module!,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(mod)},
           job, mod)
    emit_exception_flag!(mod)
end

function GPUCompiler.link_libraries!(job::CUDACompilerJob, mod::LLVM.Module,
                                     undefined_fns::Vector{String})
    invoke(GPUCompiler.link_libraries!,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(mod), typeof(undefined_fns)},
           job, mod, undefined_fns)
    link_libdevice!(mod, job.target.cap, undefined_fns)
end
