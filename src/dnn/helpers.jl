# For low level cudnn functions that require a pointer to a number
cptr(x,a::CuArray{Float64})=Float64[x]
cptr(x,a::CuArray{Float32})=Float32[x]
cptr(x,a::CuArray{Float16})=Float32[x]

# Conversion between Julia and CUDNN datatypes
cudnnDataType(::CuArray{Float16})=CUDNN_DATA_HALF
cudnnDataType(::CuArray{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::CuArray{Float64})=CUDNN_DATA_DOUBLE
cudnnDataType(::Type{Float16})=CUDNN_DATA_HALF
cudnnDataType(::Type{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::Type{Float64})=CUDNN_DATA_DOUBLE
juliaDataType(a)=(a==CUDNN_DATA_HALF ? Float16 :
                  a==CUDNN_DATA_FLOAT ? Float32 :
                  a==CUDNN_DATA_DOUBLE ? Float64 : error())

# Descriptors

mutable struct TD; ptr; end

free(td::TD) = cudnnDestroyTensorDescriptor(td.ptr)

Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TD)=td.ptr

function TD(a::CuArray, dims=ndims(a))
    (sz, st) = tensorsize(a, dims)
    d = cudnnTensorDescriptor_t[0]
    cudnnCreateTensorDescriptor(d)
    cudnnSetTensorNdDescriptor(d[1], cudnnDataType(a), length(sz), sz, st)
    this = TD(d[1])
    finalizer(this, free)
    return this
end

function tensorsize(a, dims)
    sz = Cint[reverse(size(a))...]
    st = Cint[reverse(strides(a))...]
    if length(sz) == 1 < dims
        unshift!(sz, 1)
        unshift!(st, 1)
    end
    while length(sz) < dims
        push!(sz, 1)
        push!(st, 1)
    end
    while length(sz) > dims
        d = pop!(sz)
        sz[length(sz)] *= d
        pop!(st)
        st[length(st)] = 1
    end
    (sz, st)
end
