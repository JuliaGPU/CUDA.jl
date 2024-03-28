# compatibility with EnzymeCore

module EnzymeCoreExt

using CUDA
import CUDA: GPUCompiler, CUDABackend

isdefined(Base, :get_extension) ? (import EnzymeCore) : (import ..EnzymeCore)

function EnzymeCore.compiler_job_from_backend(::CUDABackend, @nospecialize(F::Type), @nospecialize(TT::Type))
    mi = GPUCompiler.methodinstance(F, TT)
    return GPUCompiler.CompilerJob(mi, CUDA.compiler_config(CUDA.device()))
end

end # module

