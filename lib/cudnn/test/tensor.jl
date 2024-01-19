using cuDNN:
    cudnnTensorDescriptor,
    cudnnCreateTensorDescriptor,
    cudnnTensorDescriptor_t,
    cudnnFilterDescriptor,
    cudnnDataType,
    cudnnDataType_t,
    CUDNN_TENSOR_NCHW,
    CUDNN_STATUS_SUCCESS

x = CUDA.rand(1,1,1,2)

TD = cudnnTensorDescriptor
FD = cudnnFilterDescriptor
DT = cudnnDataType

@test TD(x) isa TD
@test TD(CUDNN_TENSOR_NCHW, DT(eltype(x)), Cint(ndims(x)), Cint[reverse(size(x))...]) isa TD
td = TD(x)
@test TD(td.ptr) isa TD
@test Base.unsafe_convert(Ptr, TD(td.ptr)) isa Ptr

@test FD(x) isa FD
@test FD(DT(eltype(x)),CUDNN_TENSOR_NCHW,Cint(ndims(x)),Cint[reverse(size(x))...]) isa FD
fd = FD(x)
@test FD(fd.ptr) isa FD
@test Base.unsafe_convert(Ptr, FD(fd.ptr)) isa Ptr

@test DT(Float32) isa cudnnDataType_t

@test cudnnCreateTensorDescriptor(Ref{cudnnTensorDescriptor_t}(C_NULL)) isa Nothing
