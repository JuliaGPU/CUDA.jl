# descriptor

mutable struct FilterDesc
  ptr::cudnnFilterDescriptor_t
end

unsafe_free!(fd::FilterDesc) = cudnnDestroyFilterDescriptor(fd.ptr)

Base.unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FilterDesc) = fd.ptr

function createFilterDesc()
  d = Ref{cudnnFilterDescriptor_t}()
  cudnnCreateFilterDescriptor(d)
  return d[]
end

function FilterDesc(T::Type, size::Tuple; format = CUDNN_TENSOR_NCHW)
    # The only difference of a FilterDescriptor is no strides.
    sz = Cint.(size) |> reverse |> collect
    d = createFilterDesc()
    version() >= v"5" ?
        cudnnSetFilterNdDescriptor(d, cudnnDataType(T), format, length(sz), sz) :
    version() >= v"4" ?
        cudnnSetFilterNdDescriptor_v4(d, cudnnDataType(T), format, length(sz), sz) :
        cudnnSetFilterNdDescriptor(d, cudnnDataType(T), length(sz), sz)
    this = FilterDesc(d)
    finalizer(unsafe_free!, this)
    return this
end

FilterDesc(a::CuArray; format = CUDNN_TENSOR_NCHW) = FilterDesc(eltype(a), size(a), format = format)

function Base.size(f::FilterDesc)
  typ = Ref{Cuint}()
  format = Ref{Cuint}()
  ndims = Ref{Cint}()
  dims = Vector{Cint}(undef, 8)
  cudnnGetFilterNdDescriptor(f, 8, typ, format, ndims, dims)
  @assert ndims[] ≤ 8
  return (dims[1:ndims[]]...,) |> reverse
end
