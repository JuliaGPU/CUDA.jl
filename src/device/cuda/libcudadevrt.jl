# wrappers for the libcudadevrt library
#
# The libcudadevrt library is a collection of PTX bitcode functions that implement parts of
# the CUDA API for execution on the device, such as device synchronization primitives,
# dynamic kernel APIs, etc.

import CUDAdrv: CuDim3, CuStream_t

const cudaError_t = Cint
const cudaStream_t = CUDAdrv.CuStream_t

# device-side counterpart of CUDAdrv.launch
@inline function launch(f::Ptr{Cvoid}, blocks::CuDim, threads::CuDim,
                        shmem::Int, stream::CuStream,
                        args...)
    blocks = CuDim3(blocks)
    threads = CuDim3(threads)

    buf = parameter_buffer(f, blocks, threads, shmem, args...)

    ccall("extern cudaLaunchDeviceV2", llvmcall, cudaError_t,
          (Ptr{Cvoid}, cudaStream_t),
          buf, stream)

    return
end

@generated function parameter_buffer(f::Ptr{Cvoid}, blocks::CuDim3, threads::CuDim3,
                                     shmem::Int, args...)
    # allocate a buffer
    ex = quote
        buf = ccall("extern cudaGetParameterBufferV2", llvmcall, Ptr{Cvoid},
                    (Ptr{Cvoid}, CuDim3, CuDim3, Cuint),
                    f, blocks, threads, shmem)
    end

    # store the parameters
    #
    # > Each individual parameter placed in the parameter buffer is required to be aligned.
    # > That is, each parameter must be placed at the n-th byte in the parameter buffer,
    # > where n is the smallest multiple of the parameter size that is greater than the
    # > offset of the last byte taken by the preceding parameter. The maximum size of the
    # > parameter buffer is 4KB.
    offset = 0
    for i in 1:length(args)
        buf_index = Base.ceil(Int, offset / sizeof(args[i])) + 1
        offset = buf_index * sizeof(args[i])
        push!(ex.args, :(
            unsafe_store!(Base.unsafe_convert(Ptr{$(args[i])}, buf), args[$i], $buf_index)
        ))
    end

    push!(ex.args, :(return buf))

    return ex
end

"""
    synchronize()

Wait for the device to finish. This is the device side version,
and should not be called from the host. 

`synchronize` acts as a synchronization point for
child grids in the context of dynamic parallelism.
"""
@inline synchronize() = ccall("extern cudaDeviceSynchronize", llvmcall, Cint, ())
