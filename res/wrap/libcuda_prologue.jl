# CUstream is defined as Ptr{CUstream_st}, but CUDA's headers contain aliases like
# CUstream(0x01) which cannot be directly converted to a Julia Ptr; so add a method:
mutable struct CUstream_st end
const CUstream = Ptr{CUstream_st}
CUstream(x::UInt8) = CUstream(Int(x))

# use our pointer types where possible
const CUdeviceptr = CuPtr{Cvoid}
const CUarray = CuArrayPtr{Cvoid}

# provide aliases for OpenGL-interop
const GLuint = Cuint
const GLenum = Cuint

@inline function initialize_context()
    prepare_cuda_state()
    return
end

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == ERROR_OUT_OF_MEMORY
        throw(OutOfGPUMemoryError())
    else
        throw(CuError(res))
    end
end

@inline function check(f)
    res = f()
    if res != SUCCESS
        throw_api_error(res)
    end

    return
end
