using cuDNN:
    backend_deviceprop,
    backend_tensor,
    cudnnBackendDescriptor,
    cudnnBackendDescriptor_t,
    cudnnDataType_t,
    getattr,
    getattr_int64,
    unsafe_destroy!,
    CUDNN_ATTR_TENSOR_DATA_TYPE,
    CUDNN_ATTR_TENSOR_DIMENSIONS,
    CUDNN_ATTR_TENSOR_STRIDES,
    CUDNN_ATTR_TENSOR_UNIQUE_ID,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR,
    CUDNN_DATA_FLOAT,
    CUDNN_TYPE_DATA_TYPE,
    CUDNN_TYPE_INT64

d = cudnnBackendDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR)
@test d.ptr != C_NULL
@test Base.unsafe_convert(cudnnBackendDescriptor_t, d) == d.ptr
unsafe_destroy!(d)
@test d.ptr == C_NULL
@test unsafe_destroy!(d) === nothing
@test Base.finalize(d) === nothing

t = backend_tensor(uid=42,
                   dims=Int64[1, 2, 3, 4],
                   strides=Int64[24, 12, 4, 1],
                   dtype=CUDNN_DATA_FLOAT,
                   alignment=16)
@test getattr_int64(t, CUDNN_ATTR_TENSOR_UNIQUE_ID) == 42
@test only(getattr(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE,
                   cudnnDataType_t, 1)) == CUDNN_DATA_FLOAT
@test getattr(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, Int64, 4) ==
      Int64[1, 2, 3, 4]
@test getattr(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, Int64, 4) ==
      Int64[24, 12, 4, 1]

deviceprop = backend_deviceprop()
@test deviceprop.ptr != C_NULL
