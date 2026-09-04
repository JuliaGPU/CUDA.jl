export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
end

Base.convert(::Type{cudnnStatus_t}, err::CUDNNError) = err.code

# statuses that a library without device code for the current GPU reports: cuDNN loads
# and initializes fine, and only fails once it tries to launch a kernel
const arch_statuses = (CUDNN_STATUS_ARCH_MISMATCH,
                       CUDNN_STATUS_EXECUTION_FAILED,
                       CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER,
                       CUDNN_STATUS_EXECUTION_FAILED_CUBLAS,
                       CUDNN_STATUS_EXECUTION_FAILED_CUDART,
                       CUDNN_STATUS_EXECUTION_FAILED_CURAND)

function Base.showerror(io::IO, err::CUDNNError)
    print(io, "CUDNNError: ", name(err), " (code $(reinterpret(Int32, err.code)))")
    if err.code in arch_statuses
        CUDACore.explain_unsupported_device(io)
    end
    return
end

name(err::CUDNNError) = unsafe_string(cudnnGetErrorString(err))
