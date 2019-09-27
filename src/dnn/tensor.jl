# descriptor

mutable struct TensorDesc
    ptr::cudnnTensorDescriptor_t
end

unsafe_free!(td::TensorDesc) = cudnnDestroyTensorDescriptor(td.ptr)

Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TensorDesc) = td.ptr

function TensorDesc(T::Type, size::NTuple{N,Integer}, strides::NTuple{N,Integer} = tuple_strides(size)) where N
    sz = Cint.(size) |> reverse |> collect
    st = Cint.(strides) |> reverse |> collect
    d = Ref{cudnnTensorDescriptor_t}()
    cudnnCreateTensorDescriptor(d)
    cudnnSetTensorNdDescriptor(d[], cudnnDataType(T), length(sz), sz, st)
    this = TensorDesc(d[])
    finalizer(unsafe_free!, this)
    return this
end

TensorDesc(a::CuArray) = TensorDesc(eltype(a), size(a), strides(a))


# wrappers

function cudnnAddTensor(A::CuArray{T,N}, C::CuArray{T,N}; alpha=1,
                        beta=1) where {T,N}
    aDesc = TensorDesc(A)
    cDesc = TensorDesc(C)
    cudnnAddTensor(handle(), Ref(T(alpha)), aDesc, A, Ref(T(beta)), cDesc, C)
    return C
end
