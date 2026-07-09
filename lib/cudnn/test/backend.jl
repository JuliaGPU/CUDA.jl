using cuDNN:
    backend_deviceprop,
    backend_convolution_descriptor,
    backend_resample_descriptor,
    convolution_data_backward_operation,
    convolution_filter_backward_operation,
    convolution_forward_operation,
    matmul_descriptor,
    norm_backward_operation,
    norm_forward_operation,
    pointwise_descriptor,
    reduction_descriptor,
    backend_tensor,
    diagonal_band_mask_operation,
    operation_graph,
    pointwise_operation,
    resample_backward_operation,
    resample_forward_operation,
    cudnnBackendDescriptor,
    cudnnBackendDescriptor_t,
    cudnnBackendOperationGraphMode_t,
    cudnnDataType_t,
    cudnnResampleMode_t,
    getattr,
    getattr_int64,
    make_descriptor,
    unsafe_destroy!,
    CUDNN_ATTR_OPERATIONGRAPH_MODE,
    CUDNN_ATTR_POINTWISE_MATH_PREC,
    CUDNN_ATTR_POINTWISE_MODE,
    CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
    CUDNN_ATTR_RESAMPLE_MODE,
    CUDNN_ATTR_TENSOR_DATA_TYPE,
    CUDNN_ATTR_TENSOR_DIMENSIONS,
    CUDNN_ATTR_TENSOR_STRIDES,
    CUDNN_ATTR_TENSOR_UNIQUE_ID,
    CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR,
    CUDNN_BATCH_NORM,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT,
    CUDNN_DATA_INT8,
    CUDNN_OPERATIONGRAPH_MODE_GENERIC_POINTWISE_FUSION,
    CUDNN_NORM_FWD_TRAINING,
    CUDNN_POINTWISE_CMP_GE,
    CUDNN_POINTWISE_RELU_FWD,
    CUDNN_REDUCE_TENSOR_ADD,
    CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
    CUDNN_RESAMPLE_MAXPOOL,
    CUDNN_TYPE_RESAMPLE_MODE,
    CUDNN_TYPE_OPERATIONGRAPH_MODE,
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
@test only(getattr(t, CUDNN_ATTR_TENSOR_DATA_TYPE, cudnnDataType_t, 1)) ==
      CUDNN_DATA_FLOAT

pw = make_descriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR) do d
    cuDNN.setattr!(d, CUDNN_ATTR_POINTWISE_MODE, CUDNN_POINTWISE_RELU_FWD)
    cuDNN.setattr!(d, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_DATA_FLOAT)
    cuDNN.setattr!(d, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, 0.0f0)
end
@test pw.ptr != C_NULL

pw2 = pointwise_descriptor(mode=CUDNN_POINTWISE_RELU_FWD, compute_type=CUDNN_DATA_FLOAT)
@test pw2.ptr != C_NULL

mm = matmul_descriptor(compute_type=CUDNN_DATA_FLOAT)
@test mm.ptr != C_NULL

red = reduction_descriptor(mode=CUDNN_REDUCE_TENSOR_ADD, compute_type=CUDNN_DATA_FLOAT)
@test red.ptr != C_NULL

score = backend_tensor(uid=43,
                       dims=Int64[1, 2, 3, 4],
                       strides=Int64[24, 12, 4, 1],
                       dtype=CUDNN_DATA_FLOAT,
                       alignment=16,
                       is_virtual=true)
fill = backend_tensor(uid=44,
                      dims=Int64[1, 1, 1, 1],
                      strides=Int64[1, 1, 1, 1],
                      dtype=CUDNN_DATA_FLOAT,
                      alignment=4,
                      by_value=true)
masked = backend_tensor(uid=45,
                        dims=Int64[1, 2, 3, 4],
                        strides=Int64[24, 12, 4, 1],
                        dtype=CUDNN_DATA_FLOAT,
                        alignment=16,
                        is_virtual=true)
diag = diagonal_band_mask_operation(score, fill, masked;
                                    comparison_mode=CUDNN_POINTWISE_CMP_GE)
@test diag.ptr != C_NULL

conv = backend_convolution_descriptor(compute_type=CUDNN_DATA_FLOAT,
                                      mode=CUDNN_CROSS_CORRELATION,
                                      pre_padding=Int64[1, 1],
                                      post_padding=Int64[1, 1],
                                      dilation=Int64[1, 1],
                                      stride=Int64[1, 1])
x = backend_tensor(uid=50, dims=Int64[2, 3, 5, 5], strides=Int64[75, 25, 5, 1],
                   dtype=CUDNN_DATA_FLOAT, alignment=16)
w = backend_tensor(uid=51, dims=Int64[4, 3, 3, 3], strides=Int64[27, 9, 3, 1],
                   dtype=CUDNN_DATA_FLOAT, alignment=16)
y = backend_tensor(uid=52, dims=Int64[2, 4, 5, 5], strides=Int64[100, 25, 5, 1],
                   dtype=CUDNN_DATA_FLOAT, alignment=16)
@test convolution_forward_operation(conv, x, w, y).ptr != C_NULL
@test convolution_data_backward_operation(conv, w, y, x).ptr != C_NULL
@test convolution_filter_backward_operation(conv, x, y, w).ptr != C_NULL

pool = backend_resample_descriptor(mode=CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
                                   compute_type=CUDNN_DATA_FLOAT,
                                   window=Int64[2, 2],
                                   pre_padding=Int64[0, 0],
                                   post_padding=Int64[0, 0],
                                   stride=Int64[2, 2])
@test only(getattr(pool, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE,
                   cudnnResampleMode_t, 1)) == CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING
px = backend_tensor(uid=60, dims=Int64[2, 3, 4, 4], strides=Int64[48, 16, 4, 1],
                    dtype=CUDNN_DATA_FLOAT, alignment=16)
py = backend_tensor(uid=61, dims=Int64[2, 3, 2, 2], strides=Int64[12, 4, 2, 1],
                    dtype=CUDNN_DATA_FLOAT, alignment=16)
@test resample_forward_operation(pool, px, py).ptr != C_NULL
@test resample_backward_operation(pool, px, py).ptr != C_NULL

maxpool = backend_resample_descriptor(mode=CUDNN_RESAMPLE_MAXPOOL,
                                      compute_type=CUDNN_DATA_FLOAT,
                                      window=Int64[2, 2],
                                      pre_padding=Int64[0, 0],
                                      post_padding=Int64[0, 0],
                                      stride=Int64[2, 2])
idx = backend_tensor(uid=62, dims=Int64[2, 3, 2, 2], strides=Int64[12, 4, 2, 1],
                     dtype=CUDNN_DATA_INT8, alignment=16)
@test resample_forward_operation(maxpool, px, py; index=idx).ptr != C_NULL
@test resample_backward_operation(maxpool, px, py; index=idx).ptr != C_NULL

bnx = backend_tensor(uid=70, dims=Int64[2, 3, 4, 4], strides=Int64[48, 16, 4, 1],
                     dtype=CUDNN_DATA_FLOAT, alignment=16)
bny = backend_tensor(uid=71, dims=Int64[2, 3, 4, 4], strides=Int64[48, 16, 4, 1],
                     dtype=CUDNN_DATA_FLOAT, alignment=16)
bnparam = backend_tensor(uid=72, dims=Int64[1, 3, 1, 1], strides=Int64[3, 1, 1, 1],
                         dtype=CUDNN_DATA_FLOAT, alignment=16)
eps = backend_tensor(uid=73, dims=Int64[1, 1, 1, 1], strides=Int64[1, 1, 1, 1],
                     dtype=CUDNN_DATA_FLOAT, alignment=4, by_value=true)
@test norm_forward_operation(mode=CUDNN_BATCH_NORM, phase=CUDNN_NORM_FWD_TRAINING,
                             x=bnx, mean=bnparam, inv_variance=bnparam,
                             scale=bnparam, bias=bnparam, epsilon=eps,
                             y=bny).ptr != C_NULL
@test norm_backward_operation(mode=CUDNN_BATCH_NORM, x=bnx, mean=bnparam,
                              inv_variance=bnparam, dy=bny, scale=bnparam,
                              dscale=bnparam, dbias=bnparam, dx=bnx).ptr != C_NULL

relu = pointwise_operation(pw2, score, masked)
opgraph = operation_graph([relu]; mode=CUDNN_OPERATIONGRAPH_MODE_GENERIC_POINTWISE_FUSION)
@test only(getattr(opgraph, CUDNN_ATTR_OPERATIONGRAPH_MODE,
                   CUDNN_TYPE_OPERATIONGRAPH_MODE,
                   cudnnBackendOperationGraphMode_t, 1)) ==
      CUDNN_OPERATIONGRAPH_MODE_GENERIC_POINTWISE_FUSION

deviceprop = backend_deviceprop()
@test deviceprop.ptr != C_NULL
