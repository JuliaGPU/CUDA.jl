# E. Dynamic Parallelism

import CUDAdrv: CuDim3, CUstream
const cudaStream_t = CUDAdrv.CUstream

# device-side counterpart of CUDAdrv.launch
@inline function launch(f, blocks, threads, shmem, stream, args...)
    blocks = CuDim3(blocks)
    threads = CuDim3(threads)

    buf = parameter_buffer(f, blocks, threads, shmem, args...)
    cudaLaunchDeviceV2(buf, stream)

    return
end

@generated function parameter_buffer(f, blocks::CuDim3, threads::CuDim3, shmem, args...)
    # allocate a buffer
    ex = quote
        Base.@_inline_meta
        buf = cudaGetParameterBufferV2(f, blocks, threads, shmem)
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
        T = args[i]
        align = sizeof(T)
        buf_index = Base.ceil(Int, offset / align) + 1
        offset = buf_index * align
        ptr = :(Base.unsafe_convert(Ptr{$T}, buf))
        push!(ex.args, :(
            Base.pointerset($ptr, args[$i], $buf_index, $align)
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
synchronize() = cudaDeviceSynchronize()
