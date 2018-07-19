import GPUArrays

Base.similar(::Type{<:CuArray}, ::Type{T}, size::Base.Dims{N}) where {T, N} = CuArray{T, N}(size)

#Abstract GPU interface
struct CuKernelState end

@inline function GPUArrays.LocalMemory(::CuKernelState, ::Type{T}, ::Val{N}, ::Val{C}) where {T, N, C}
    CUDAnative.generate_static_shmem(Val{C}, T, Val{N})
end

(::Type{GPUArrays.AbstractDeviceArray})(A::CUDAnative.CuDeviceArray, shape) = CUDAnative.CuDeviceArray(shape, pointer(A))


@inline GPUArrays.synchronize_threads(::CuKernelState) = CUDAnative.sync_threads()

GPUArrays.blas_module(::CuArray) = CuArrays.BLAS
GPUArrays.blasbuffer(x::CuArray) = x

"""
Blocks until all operations are finished on `A`
"""
function GPUArrays.synchronize(A::CuArray)
    # fallback is a noop, for backends not needing synchronization. This
    # makes it easier to write generic code that also works for AbstractArrays
    CUDAdrv.synchronize(CUDAnative.default_context[])
end

for (i, sym) in enumerate((:x, :y, :z))
    for (f, fcu) in (
            (:blockidx, :blockIdx),
            (:blockdim, :blockDim),
            (:threadidx, :threadIdx),
            (:griddim, :gridDim)
        )
        fname = Symbol(string(f, '_', sym))
        cufun = Symbol(string(fcu, '_', sym))
        @eval GPUArrays.$fname(::CuKernelState)::Cuint = CUDAnative.$cufun()
    end
end

# devices() = CUDAdrv.devices()
GPUArrays.device(A::CuArray) = CUDAdrv.device(CUDAdrv.CuCurrentContext())
GPUArrays.is_gpu(dev::CUDAdrv.CuDevice) = true
GPUArrays.name(dev::CUDAdrv.CuDevice) = string("CU ", CUDAdrv.name(dev))
GPUArrays.threads(dev::CUDAdrv.CuDevice) = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)

function GPUArrays.blocks(dev::CUDAdrv.CuDevice)
    (
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_X),
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Y),
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Z),
    )
end

GPUArrays.free_global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.Mem.info()[1]
GPUArrays.global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.totalmem(dev)
GPUArrays.local_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.attribute(dev, CUDAdrv.TOTAL_CONSTANT_MEMORY)


function GPUArrays._gpu_call(f, A::CuArray, args::Tuple, blocks_threads::Tuple{T, T}) where T <: NTuple{N, Integer} where N
    blk, thr = blocks_threads
    @cuda blocks=blk threads=thr f(CuKernelState(), args...)
end

# Save reinterpret and reshape implementation use this in GPUArrays
function GPUArrays.unsafe_reinterpret(::Type{T}, A::CuArray{ET}, size::NTuple{N, Integer}) where {T, ET, N}
    CuArray{T, N}(A.buf, size)
end

# Additional copy methods

function Base.copyto!(
        dest::Array{T}, d_offset::Integer,
        source::CuArray{T}, s_offset::Integer, amount::Integer
    ) where T
    amount == 0 && return dest
    d_offset = d_offset
    s_offset = s_offset - 1
    buf = Mem.view(source.buf, s_offset*sizeof(T))
    CUDAdrv.Mem.download!(Ref(dest, d_offset), buf, sizeof(T) * (amount))
    dest
end

function Base.copyto!(
        dest::CuArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    ) where T
    amount == 0 && return dest
    d_offset = d_offset - 1
    s_offset = s_offset
    buf = Mem.view(dest.buf, s_offset*sizeof(T))
    CUDAdrv.Mem.upload!(buf, Ref(source, s_offset), sizeof(T) * (amount))
    dest
end

function Base.copyto!(
        dest::CuArray{T}, d_offset::Integer,
        source::CuArray{T}, s_offset::Integer, amount::Integer
    ) where T
    d_offset = d_offset - 1
    s_offset = s_offset - 1
    d_ptr = dest.ptr
    s_ptr = source.ptr
    dptr = d_ptr + (sizeof(T) * d_offset)
    sptr = s_ptr + (sizeof(T) * s_offset)
    CUDAdrv.Mem.transfer!(sptr, dptr, sizeof(T) * (amount))
    dest
end
