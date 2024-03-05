# compatibility with EnzymeCore

module EnzymeCoreExt

using CUDA: CUDABackend

isdefined(Base, :get_extension) ? (import EnzymeCore) : (import ..EnzymeCore)

function EnzymeCore.parent_job_for_tape_type(::CUDABackend)
    mi = CUDA.methodinstance(typeof(()->return), Tuple{})
    return CUDA.CompilerJob(mi, CUDA.compiler_config(CUDA.device()))
end

end # module

