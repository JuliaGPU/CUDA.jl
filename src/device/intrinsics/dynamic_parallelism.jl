# E. Dynamic Parallelism

## error handling

# TODO: generalize for all uses of cudadevrt

struct CuDeviceError <: Exception
    code::cudaError
end

Base.convert(::Type{cudaError_t}, err::CuDeviceError) = err.code

name(err::CuDeviceError) = cudaGetErrorName(err)

description(err::CuDeviceError) = cudaGetErrorString(err)

@noinline function throw_device_cuerror(err::CuDeviceError)
    # the exception won't be rendered on the host, so print some details here already
    @cuprintf("ERROR: a CUDA error was thrown during kernel execution: %s (code %d), %s)",
              description(err), Int32(err.code), name(err))
    throw(err)
end

macro check_status(ex)
    quote
        res = $(esc(ex))
        if res != cudaSuccess
            throw_device_cuerror(CuDeviceError(res))
        end
    end
end


## streams

export CuDeviceStream

struct CuDeviceStream
    handle::cudaStream_t

    function CuDeviceStream(flags=cudaStreamNonBlocking)
        handle_ref = Ref{cudaStream_t}()
        @check_status cudaStreamCreateWithFlags(handle_ref, flags)
        return new(handle_ref[])
    end

    global CuDefaultDeviceStream() = new(convert(cudaStream_t, C_NULL))
end

Base.unsafe_convert(::Type{cudaStream_t}, s::CuDeviceStream) = s.handle

function unsafe_destroy!(s::CuDeviceStream)
    @check_status cudaStreamDestroy(s)
    return
end


## execution

struct CuDeviceFunction
    ptr::Ptr{Cvoid}
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, fun::CuDeviceFunction) = fun.ptr

function launch(f::CuDeviceFunction, args::Vararg{Any,N}; blocks::CuDim=1, threads::CuDim=1,
                shmem::Integer=0, stream::CuDeviceStream=CuDefaultDeviceStream()) where {N}
    blockdim = CuDim3(blocks)
    threaddim = CuDim3(threads)

    buf = parameter_buffer(f, blockdim, threaddim, shmem, args...)
    @check_status cudaLaunchDeviceV2(buf, stream)

    return
end

@inline @generated function parameter_buffer(f::CuDeviceFunction, blocks, threads, shmem, args...)
    # allocate a buffer
    ex = quote
        buf = cudaGetParameterBufferV2(f, blocks, threads, shmem)
        ptr = Base.unsafe_convert(Ptr{UInt32}, buf)
    end

    # store the parameters
    #
    # D.3.2.2. Parameter Buffer Layout
    # > Each individual parameter placed in the parameter buffer is required to be aligned.
    # > That is, each parameter must be placed at the n-th byte in the parameter buffer,
    # > where n is the smallest multiple of the parameter size that is greater than the
    # > offset of the last byte taken by the preceding parameter. The maximum size of the
    # > parameter buffer is 4KB.
    #
    # NOTE: the above seems wrong, and we should use the parameter alignment, not its size.
    last_offset = 0
    for i in 1:length(args)
        T = args[i]
        align = Base.datatype_alignment(T)
        offset = Base.cld(last_offset, align) * align
        push!(ex.args, :(
            Base.pointerset(convert(Ptr{$T}, ptr+$offset), args[$i], 1, $align)
        ))
        last_offset = offset + sizeof(T)
    end

    push!(ex.args, :(return buf))

    return ex
end


## synchronization

@device_override device_synchronize() = @check_status cudaDeviceSynchronize()
@doc """
On the device, `device_synchronize` acts as a synchronization point for child grids in the
context of dynamic parallelism.
""" device_synchronize
