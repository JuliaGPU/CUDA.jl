# Julia wrapper for header: cuda_device_runtime_api.h
# Automatically generated using Clang.jl

# Not fully automated, since Clang.jl can't handle the `extern "C"`, and we
# change the `ccall` invocation to refer to a manually-provided library (`extern ...`).
# Only the entry points used by the dynamic parallelism support are wrapped.

@device_functions begin
function cudaDeviceSynchronize()
    ccall("extern cudaDeviceSynchronize", llvmcall, cudaError_t, ())
end

function cudaGetLastError()
    ccall("extern cudaGetLastError", llvmcall, cudaError_t, ())
end

function cudaGetErrorString(error)
    ccall("extern cudaGetErrorString", llvmcall, Cstring,
          (cudaError_t,),
          error)
end

function cudaGetErrorName(error)
    ccall("extern cudaGetErrorName", llvmcall, Cstring,
          (cudaError_t,),
          error)
end

function cudaStreamCreateWithFlags(pStream, flags)
    ccall("extern cudaStreamCreateWithFlags", llvmcall, cudaError_t,
        (Ptr{cudaStream_t}, UInt32),
        pStream, flags)
end

function cudaStreamDestroy(stream)
    ccall("extern cudaStreamDestroy", llvmcall, cudaError_t,
        (cudaStream_t,),
        stream)
end

function cudaGetParameterBufferV2(func, gridDimension, blockDimension, sharedMemSize)
    ccall("extern cudaGetParameterBufferV2", llvmcall, Ptr{Cvoid},
        (Ptr{Cvoid}, dim3, dim3, UInt32),
        func, gridDimension, blockDimension, sharedMemSize)
end

function cudaLaunchDeviceV2(parameterBuffer, stream)
    ccall("extern cudaLaunchDeviceV2", llvmcall, cudaError_t,
        (Ptr{Cvoid}, cudaStream_t),
        parameterBuffer, stream)
end
end
