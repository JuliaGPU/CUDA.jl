# E. Dynamic Parallelism

import CUDAdrv: CuDim3, CUstream
const cudaStream_t = CUDAdrv.CUstream

if VERSION >= v"1.2.0-DEV.512"
    @inline cudaLaunchDevice(buf::Ptr{Cvoid}, stream::CuStream) =
        ccall("extern cudaLaunchDeviceV2", llvmcall, cudaError_t,
              (Ptr{Cvoid}, cudaStream_t),
              buf, stream)
else
    import Base.Sys: WORD_SIZE
    # declare i32 @cudaLaunchDeviceV2(i8*, %struct.CUstream_st*)
    @eval @inline cudaLaunchDevice(buf::Ptr{Cvoid}, stream::CuStream) =
            Base.llvmcall(
                $("declare i32 @cudaLaunchDeviceV2(i8*, i8*)",
                  "%buf = inttoptr i$WORD_SIZE %0 to i8*
                   %stream = inttoptr i$WORD_SIZE %1 to i8*
                   %rv = call i32 @cudaLaunchDeviceV2(i8* %buf, i8* %stream)
                   ret i32 %rv"), cudaError_t,
                Tuple{Ptr{Cvoid}, cudaStream_t},
                buf, Base.unsafe_convert(cudaStream_t, stream))
end

# device-side counterpart of CUDAdrv.launch
@inline function launch(f::Ptr{Cvoid}, blocks::CuDim, threads::CuDim,
                        shmem::Int, stream::CuStream,
                        args...)
    blocks = CuDim3(blocks)
    threads = CuDim3(threads)

    buf = parameter_buffer(f, blocks, threads, shmem, args...)
    cudaLaunchDevice(buf, stream)

    return
end

if VERSION >= v"1.2.0-DEV.512"
    @inline cudaGetParameterBuffer(f::Ptr{Cvoid}, blocks::CuDim3, threads::CuDim3, shmem::Integer) =
        ccall("extern cudaGetParameterBufferV2", llvmcall, Ptr{Cvoid},
                            (Ptr{Cvoid}, CuDim3, CuDim3, Cuint),
                            f, blocks, threads, shmem)
else
    @inline cudaGetParameterBuffer(f::Ptr{Cvoid}, blocks::CuDim3, threads::CuDim3, shmem::Integer) =
        cudaGetParameterBuffer(f,
                               blocks.x, blocks.y, blocks.z,
                               threads.x, threads.y, threads.z,
                               convert(Cuint, shmem))
    # declare i8* @cudaGetParameterBufferV2(i8*, %struct.dim3, %struct.dim3, i32)
    @eval @inline cudaGetParameterBuffer(f::Ptr{Cvoid},
                                         blocks_x::Cuint, blocks_y::Cuint, blocks_z::Cuint,
                                         threads_x::Cuint, threads_y::Cuint, threads_z::Cuint,
                                         shmem::Cuint) =
            Base.llvmcall(
                $("declare i8* @cudaGetParameterBufferV2(i8*, {i32,i32,i32}, {i32,i32,i32}, i32)",
                  "%f = inttoptr i$WORD_SIZE %0 to i8*
                   %blocks.x = insertvalue { i32, i32, i32 } undef, i32 %1, 0
                   %blocks.y = insertvalue { i32, i32, i32 } %blocks.x, i32 %2, 1
                   %blocks.z = insertvalue { i32, i32, i32 } %blocks.y, i32 %3, 2
                   %threads.x = insertvalue { i32, i32, i32 } undef, i32 %4, 0
                   %threads.y = insertvalue { i32, i32, i32 } %threads.x, i32 %5, 1
                   %threads.z = insertvalue { i32, i32, i32 } %threads.y, i32 %6, 2
                   %rv = call i8* @cudaGetParameterBufferV2(i8* %f, {i32,i32,i32} %blocks.z, {i32,i32,i32} %threads.z, i32 %7)
                   %buf = ptrtoint i8* %rv to i$WORD_SIZE
                   ret i$WORD_SIZE %buf"), Ptr{Cvoid},
                Tuple{Ptr{Cvoid}, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint},
                f, blocks_x, blocks_y, blocks_z, threads_x, threads_y, threads_z, shmem)
end

@generated function parameter_buffer(f, blocks::CuDim3, threads::CuDim3, shmem, args...)
    # allocate a buffer
    ex = quote
        Base.@_inline_meta
        buf = cudaGetParameterBuffer(f, blocks, threads, shmem)
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

if VERSION >= v"1.2.0-DEV.512"
    @inline synchronize() = ccall("extern cudaDeviceSynchronize", llvmcall, Cint, ())
else
    @eval @inline synchronize() = Base.llvmcall(
        $("declare i32 @cudaDeviceSynchronize()",
          "%rv = call i32 @cudaDeviceSynchronize()
           ret i32 %rv"), cudaError_t, Tuple{})
end

"""
    synchronize()

Wait for the device to finish. This is the device side version,
and should not be called from the host.

`synchronize` acts as a synchronization point for
child grids in the context of dynamic parallelism.
"""
synchronize
