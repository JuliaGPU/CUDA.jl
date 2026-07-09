# Thin Julia layer over the cuDNN *backend graph API* (the `cudnnBackend*` functions).
#
# Unlike the legacy descriptor API (see descriptors.jl), the backend API is fully generic:
# every object is a `cudnnBackendDescriptor_t` configured through
# `cudnnBackendSetAttribute(desc, name, type, count, ptr)`, finalized with
# `cudnnBackendFinalize`, and read back with `cudnnBackendGetAttribute`. This file provides a
# typed wrapper plus small constructor helpers for the pieces needed to build and run a fused
# graph (tensors, operation graph, engine heuristics, execution plan, variant pack). It is
# used by the graph frontend in graph/.

# A finalized-or-not backend descriptor. Owns the handle and destroys it on GC.
mutable struct cudnnBackendDescriptor
    ptr::cudnnBackendDescriptor_t
end

function cudnnBackendDescriptor(descriptorType::cudnnBackendDescriptorType_t)
    ref = Ref{cudnnBackendDescriptor_t}(C_NULL)
    cudnnBackendCreateDescriptor(descriptorType, ref)
    d = cudnnBackendDescriptor(ref[])
    finalizer(unsafe_destroy!, d)
    return d
end

Base.unsafe_convert(::Type{cudnnBackendDescriptor_t}, d::cudnnBackendDescriptor) = d.ptr

function unsafe_destroy!(d::cudnnBackendDescriptor)
    ptr = d.ptr
    ptr == C_NULL && return
    d.ptr = C_NULL
    cudnnBackendDestroyDescriptor(ptr)
    return
end

function make_descriptor(f, type::cudnnBackendDescriptorType_t)
    d = cudnnBackendDescriptor(type)
    try
        f(d)
        cudnnBackendFinalize(d)
        return d
    catch
        unsafe_destroy!(d)
        rethrow()
    end
end


# --- setattr! ---------------------------------------------------------------------------
#
# Set a backend attribute, dispatching on the Julia value type to pick the cuDNN attribute
# type, element count, and host buffer. The buffer is GC-preserved across the ccall.

# core: `buf` is a host array/Ref backing `count` contiguous elements of the attribute type.
function setattr!(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                  atype::cudnnBackendAttributeType_t, count::Integer, buf)
    GC.@preserve buf begin
        cudnnBackendSetAttribute(d.ptr, name, atype, Int64(count),
                                 convert(Ptr{Cvoid}, pointer(buf)))
    end
    return d
end

# integer dims/strides/ids/sizes
setattr!(d, name, v::Integer) = setattr!(d, name, CUDNN_TYPE_INT64, 1, Int64[v])
setattr!(d, name, v::AbstractVector{<:Integer}) =
    setattr!(d, name, CUDNN_TYPE_INT64, length(v), convert(Vector{Int64}, v))

# booleans (cuDNN CUDNN_TYPE_BOOLEAN is a 1-byte bool, matching Julia Bool)
setattr!(d, name, v::Bool) = setattr!(d, name, CUDNN_TYPE_BOOLEAN, 1, Bool[v])

# doubles (e.g. dropout probability)
setattr!(d, name, v::Float32) = setattr!(d, name, CUDNN_TYPE_FLOAT, 1, Float32[v])
setattr!(d, name, v::Float64) = setattr!(d, name, CUDNN_TYPE_DOUBLE, 1, Float64[v])

# bytes and raw pointers
setattr!(d, name, v::Char) = setattr!(d, name, CUDNN_TYPE_CHAR, 1, UInt8[UInt8(v)])
setattr!(d, name, v::Ptr) =
    setattr!(d, name, CUDNN_TYPE_VOID_PTR, 1, Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, v)])
setattr!(d, name, v::AbstractVector{<:Ptr}) =
    setattr!(d, name, CUDNN_TYPE_VOID_PTR, length(v),
             Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, x) for x in v])

backend_attribute_type(::Type{cudnnDataType_t}) = CUDNN_TYPE_DATA_TYPE
backend_attribute_type(::Type{cudnnBackendHeurMode_t}) = CUDNN_TYPE_HEUR_MODE
backend_attribute_type(::Type{cudnnPointwiseMode_t}) = CUDNN_TYPE_POINTWISE_MODE
backend_attribute_type(::Type{cudnnConvolutionMode_t}) = CUDNN_TYPE_CONVOLUTION_MODE
backend_attribute_type(::Type{cudnnNanPropagation_t}) = CUDNN_TYPE_NAN_PROPOGATION
backend_attribute_type(::Type{cudnnBackendNumericalNote_t}) = CUDNN_TYPE_NUMERICAL_NOTE
backend_attribute_type(::Type{cudnnBackendBehaviorNote_t}) = CUDNN_TYPE_BEHAVIOR_NOTE
backend_attribute_type(::Type{cudnnReduceTensorOp_t}) = CUDNN_TYPE_REDUCTION_OPERATOR_TYPE
backend_attribute_type(::Type{cudnnResampleMode_t}) = CUDNN_TYPE_RESAMPLE_MODE
backend_attribute_type(::Type{cudnnPaddingMode_t}) = CUDNN_TYPE_PADDING_MODE
backend_attribute_type(::Type{cudnnBackendNormMode_t}) = CUDNN_TYPE_NORM_MODE
backend_attribute_type(::Type{cudnnBackendNormFwdPhase_t}) = CUDNN_TYPE_NORM_FWD_PHASE
backend_attribute_type(::Type{cudnnRngDistribution_t}) = CUDNN_TYPE_RNG_DISTRIBUTION
backend_attribute_type(::Type{cudnnMoeGroupedMatmulMode_t}) = CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE
backend_attribute_type(::Type{cudnnBackendOperationGraphMode_t}) = CUDNN_TYPE_OPERATIONGRAPH_MODE
backend_attribute_type(::Type{cudnnFraction_t}) = CUDNN_TYPE_FRACTION

setattr!(d, name, v::cudnnFraction_t) =
    setattr!(d, name, CUDNN_TYPE_FRACTION, 1, cudnnFraction_t[v])
setattr!(d, name, v::AbstractVector{cudnnFraction_t}) =
    setattr!(d, name, CUDNN_TYPE_FRACTION, length(v), collect(v))
setattr!(d, name, v::T) where {T<:CEnum.Cenum} =
    setattr!(d, name, backend_attribute_type(T), 1, T[v])
setattr!(d, name, v::AbstractVector{T}) where {T<:CEnum.Cenum} =
    setattr!(d, name, backend_attribute_type(T), length(v), convert(Vector{T}, v))

# the cuDNN handle (accepts the raw handle or the `Handle` wrapper from handle())
setattr_handle!(d, name) =
    setattr!(d, name, CUDNN_TYPE_HANDLE, 1,
             cudnnHandle_t[Base.unsafe_convert(cudnnHandle_t, handle())])

# nested descriptor(s)
setattr!(d, name, v::cudnnBackendDescriptor) =
    setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, cudnnBackendDescriptor_t[v.ptr])
setattr!(d, name, v::AbstractVector{cudnnBackendDescriptor}) =
    setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, length(v),
             cudnnBackendDescriptor_t[x.ptr for x in v])


# --- getattr ----------------------------------------------------------------------------

# Read up to `maxcount` plain (non-descriptor) elements of type `T`.
function getattr(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                 atype::cudnnBackendAttributeType_t, ::Type{T}, maxcount::Integer) where {T}
    out = Vector{T}(undef, maxcount)
    n = Ref{Int64}(0)
    GC.@preserve out begin
        cudnnBackendGetAttribute(d.ptr, name, atype, Int64(maxcount), n,
                                 convert(Ptr{Cvoid}, pointer(out)))
    end
    resize!(out, n[])
    return out
end

getattr(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t, ::Type{T},
        maxcount::Integer=1) where {T} =
    getattr(d, name, backend_attribute_type(T), T, maxcount)

getattr_float32(d, name) = getattr(d, name, CUDNN_TYPE_FLOAT, Float32, 1)[]
getattr_float64(d, name) = getattr(d, name, CUDNN_TYPE_DOUBLE, Float64, 1)[]
getattr_int64(d, name) = getattr(d, name, CUDNN_TYPE_INT64, Int64, 1)[]

function getattr_count(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                       atype::cudnnBackendAttributeType_t)
    n = Ref{Int64}(0)
    cudnnBackendGetAttribute(d.ptr, name, atype, Int64(0), n, C_NULL)
    return n[]
end

# Read an array of nested descriptors. cuDNN requires the caller to pre-create the output
# descriptors; GetAttribute then populates their contents.
#
# Keep raw handles until we know how many cuDNN returned, then destroy the unused descriptors
# synchronously. Julia finalizers are a last-resort cleanup path for descriptors that escape.
function getattr_descriptors(d::cudnnBackendDescriptor, name::cudnnBackendAttributeName_t,
                             desctype::cudnnBackendDescriptorType_t, maxcount::Integer)
    maxcount == 0 && return cudnnBackendDescriptor[]
    raw = fill(cudnnBackendDescriptor_t(C_NULL), maxcount)
    n = Ref{Int64}(0)
    try
        for i in 1:maxcount
            r = Ref{cudnnBackendDescriptor_t}(C_NULL)
            cudnnBackendCreateDescriptor(desctype, r)
            raw[i] = r[]
        end
        GC.@preserve raw begin
            cudnnBackendGetAttribute(d.ptr, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, Int64(maxcount),
                                     n, convert(Ptr{Cvoid}, pointer(raw)))
        end
    catch
        for ptr in raw
            ptr == C_NULL || cudnnBackendDestroyDescriptor(ptr)
        end
        rethrow()
    end
    nreturned = min(n[], Int64(maxcount))
    for i in nreturned+1:maxcount
        cudnnBackendDestroyDescriptor(raw[i])
    end
    nreturned == 0 && return cudnnBackendDescriptor[]
    return map(1:nreturned) do i
        desc = cudnnBackendDescriptor(raw[i])
        finalizer(unsafe_destroy!, desc)
        desc
    end
end


# --- operation-node constructor helpers -------------------------------------------------

"""
    backend_tensor(; uid, dims, strides, dtype, is_virtual=false, by_value=false, alignment=16)

Create and finalize a `CUDNN_BACKEND_TENSOR_DESCRIPTOR`. `dims`/`strides` are in cuDNN order
(outermost first, innermost last; the innermost stride is typically 1).
"""
function backend_tensor(; uid::Integer, dims, strides, dtype::cudnnDataType_t,
                        is_virtual::Bool=false, by_value::Bool=false, alignment::Integer=16)
    make_descriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR) do d
        setattr!(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, Int64(uid))
        setattr!(d, CUDNN_ATTR_TENSOR_DATA_TYPE, dtype)
        setattr!(d, CUDNN_ATTR_TENSOR_DIMENSIONS, dims)
        setattr!(d, CUDNN_ATTR_TENSOR_STRIDES, strides)
        setattr!(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, Int64(alignment))
        is_virtual && setattr!(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, true)
        by_value && setattr!(d, CUDNN_ATTR_TENSOR_IS_BY_VALUE, true)
    end
end

function backend_deviceprop()
    make_descriptor(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR) do d
        setattr_handle!(d, CUDNN_ATTR_DEVICEPROP_HANDLE)
    end
end

function operation_graph(ops::AbstractVector{cudnnBackendDescriptor};
                         mode::cudnnBackendOperationGraphMode_t=CUDNN_OPERATIONGRAPH_MODE_AUTO)
    make_descriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR) do g
        setattr_handle!(g, CUDNN_ATTR_OPERATIONGRAPH_HANDLE)
        setattr!(g, CUDNN_ATTR_OPERATIONGRAPH_OPS, ops)
        mode == CUDNN_OPERATIONGRAPH_MODE_AUTO ||
            setattr!(g, CUDNN_ATTR_OPERATIONGRAPH_MODE, mode)
    end
end

function pointwise_descriptor(; mode::cudnnPointwiseMode_t, compute_type::cudnnDataType_t,
                              nan_propagation::cudnnNanPropagation_t=CUDNN_NOT_PROPAGATE_NAN,
                              relu_lower_clip::Real=0, relu_upper_clip::Real=Inf,
                              relu_lower_clip_slope::Real=0, elu_alpha::Real=1,
                              softplus_beta::Real=1, swish_beta::Real=1,
                              axis::Union{Nothing,Integer}=nothing)
    make_descriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR) do d
        setattr!(d, CUDNN_ATTR_POINTWISE_MODE, mode)
        setattr!(d, CUDNN_ATTR_POINTWISE_MATH_PREC, compute_type)
        if mode in (CUDNN_POINTWISE_RELU_FWD, CUDNN_POINTWISE_RELU_BWD)
            setattr!(d, CUDNN_ATTR_POINTWISE_NAN_PROPAGATION, nan_propagation)
            setattr!(d, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, Float64(relu_lower_clip))
            setattr!(d, CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP, Float64(relu_upper_clip))
            setattr!(d, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                     Float64(relu_lower_clip_slope))
        elseif mode in (CUDNN_POINTWISE_ELU_FWD, CUDNN_POINTWISE_ELU_BWD)
            setattr!(d, CUDNN_ATTR_POINTWISE_ELU_ALPHA, Float64(elu_alpha))
        elseif mode == CUDNN_POINTWISE_SOFTPLUS_FWD
            setattr!(d, CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, Float64(softplus_beta))
        elseif mode in (CUDNN_POINTWISE_SWISH_FWD, CUDNN_POINTWISE_SWISH_BWD)
            setattr!(d, CUDNN_ATTR_POINTWISE_SWISH_BETA, Float64(swish_beta))
        end
        axis === nothing || setattr!(d, CUDNN_ATTR_POINTWISE_AXIS, Int64(axis))
    end
end

function matmul_descriptor(; compute_type::cudnnDataType_t)
    make_descriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR) do d
        setattr!(d, CUDNN_ATTR_MATMUL_COMP_TYPE, compute_type)
    end
end

function reduction_descriptor(; mode::cudnnReduceTensorOp_t, compute_type::cudnnDataType_t,
                              deterministic::Bool=false)
    make_descriptor(CUDNN_BACKEND_REDUCTION_DESCRIPTOR) do d
        setattr!(d, CUDNN_ATTR_REDUCTION_COMP_TYPE, compute_type)
        setattr!(d, CUDNN_ATTR_REDUCTION_OPERATOR, mode)
        setattr!(d, CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC, deterministic)
    end
end

function backend_convolution_descriptor(; compute_type::cudnnDataType_t,
                                        mode::cudnnConvolutionMode_t,
                                        pre_padding, post_padding, dilation, stride)
    spatial_dims = length(pre_padding)
    length(post_padding) == spatial_dims && length(dilation) == spatial_dims &&
        length(stride) == spatial_dims ||
        throw(DimensionMismatch("convolution attributes must have matching rank"))
    make_descriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR) do d
        setattr!(d, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, compute_type)
        setattr!(d, CUDNN_ATTR_CONVOLUTION_CONV_MODE, mode)
        setattr!(d, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, spatial_dims)
        setattr!(d, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, pre_padding)
        setattr!(d, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, post_padding)
        setattr!(d, CUDNN_ATTR_CONVOLUTION_DILATIONS, dilation)
        setattr!(d, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, stride)
    end
end

fraction(x::cudnnFraction_t) = x
fraction(x::Integer) = cudnnFraction_t(Int64(x), Int64(1))
fractions(xs) = cudnnFraction_t[fraction(x) for x in xs]

function backend_resample_descriptor(; mode::cudnnResampleMode_t,
                                     compute_type::cudnnDataType_t,
                                     window, pre_padding, post_padding, stride,
                                     nan_propagation::cudnnNanPropagation_t=CUDNN_PROPAGATE_NAN,
                                     padding_mode::cudnnPaddingMode_t=CUDNN_ZERO_PAD)
    spatial_dims = length(window)
    length(pre_padding) == spatial_dims && length(post_padding) == spatial_dims &&
        length(stride) == spatial_dims ||
        throw(DimensionMismatch("resample attributes must have matching rank"))
    make_descriptor(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR) do d
        setattr!(d, CUDNN_ATTR_RESAMPLE_MODE, mode)
        setattr!(d, CUDNN_ATTR_RESAMPLE_COMP_TYPE, compute_type)
        setattr!(d, CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION, nan_propagation)
        setattr!(d, CUDNN_ATTR_RESAMPLE_PADDING_MODE, padding_mode)
        setattr!(d, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, spatial_dims)
        setattr!(d, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, fractions(window))
        setattr!(d, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, fractions(pre_padding))
        setattr!(d, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, fractions(post_padding))
        setattr!(d, CUDNN_ATTR_RESAMPLE_STRIDES, fractions(stride))
    end
end

function pointwise_operation(pwdesc::cudnnBackendDescriptor, x::cudnnBackendDescriptor,
                             y::cudnnBackendDescriptor;
                             b::Union{Nothing,cudnnBackendDescriptor}=nothing,
                             t::Union{Nothing,cudnnBackendDescriptor}=nothing,
                             alpha1::Real=1, alpha2::Real=0)
    make_descriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pwdesc)
        setattr!(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x)
        setattr!(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y)
        setattr!(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, Float32(alpha1))
        setattr!(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, Float32(alpha2))
        b === nothing || setattr!(op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, b)
        t === nothing || setattr!(op, CUDNN_ATTR_OPERATION_POINTWISE_TDESC, t)
    end
end

function matmul_operation(matdesc::cudnnBackendDescriptor, a::cudnnBackendDescriptor,
                          b::cudnnBackendDescriptor, c::cudnnBackendDescriptor)
    make_descriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_MATMUL_ADESC, a)
        setattr!(op, CUDNN_ATTR_OPERATION_MATMUL_BDESC, b)
        setattr!(op, CUDNN_ATTR_OPERATION_MATMUL_CDESC, c)
        setattr!(op, CUDNN_ATTR_OPERATION_MATMUL_DESC, matdesc)
    end
end

function reduction_operation(reddesc::cudnnBackendDescriptor, x::cudnnBackendDescriptor,
                             y::cudnnBackendDescriptor)
    make_descriptor(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_REDUCTION_DESC, reddesc)
        setattr!(op, CUDNN_ATTR_OPERATION_REDUCTION_XDESC, x)
        setattr!(op, CUDNN_ATTR_OPERATION_REDUCTION_YDESC, y)
    end
end

function convolution_forward_operation(convdesc::cudnnBackendDescriptor,
                                       x::cudnnBackendDescriptor,
                                       w::cudnnBackendDescriptor,
                                       y::cudnnBackendDescriptor;
                                       alpha::Real=1, beta::Real=0,
                                       alphabeta_type::cudnnBackendAttributeType_t=CUDNN_TYPE_FLOAT)
    make_descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, convdesc)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y)
        setattr_alphabeta!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, alpha,
                           alphabeta_type)
        setattr_alphabeta!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, beta,
                           alphabeta_type)
    end
end

function convolution_data_backward_operation(convdesc::cudnnBackendDescriptor,
                                             w::cudnnBackendDescriptor,
                                             dy::cudnnBackendDescriptor,
                                             dx::cudnnBackendDescriptor;
                                             alpha::Real=1, beta::Real=0,
                                             alphabeta_type::cudnnBackendAttributeType_t=CUDNN_TYPE_FLOAT)
    make_descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, convdesc)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, w)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, dy)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, dx)
        setattr_alphabeta!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, alpha,
                           alphabeta_type)
        setattr_alphabeta!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, beta,
                           alphabeta_type)
    end
end

function convolution_filter_backward_operation(convdesc::cudnnBackendDescriptor,
                                               x::cudnnBackendDescriptor,
                                               dy::cudnnBackendDescriptor,
                                               dw::cudnnBackendDescriptor;
                                               alpha::Real=1, beta::Real=0,
                                               alphabeta_type::cudnnBackendAttributeType_t=CUDNN_TYPE_FLOAT)
    make_descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, convdesc)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, x)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, dy)
        setattr!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, dw)
        setattr_alphabeta!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, alpha,
                           alphabeta_type)
        setattr_alphabeta!(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, beta,
                           alphabeta_type)
    end
end

function setattr_alphabeta!(d::cudnnBackendDescriptor, name, value, atype)
    if atype == CUDNN_TYPE_DOUBLE
        setattr!(d, name, CUDNN_TYPE_DOUBLE, 1, Float64[value])
    else
        setattr!(d, name, CUDNN_TYPE_FLOAT, 1, Float32[value])
    end
end

function resample_forward_operation(resampledesc::cudnnBackendDescriptor,
                                    x::cudnnBackendDescriptor,
                                    y::cudnnBackendDescriptor;
                                    index::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                    alpha::Real=1, beta::Real=0)
    make_descriptor(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, x)
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, y)
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA, Float64(alpha))
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA, Float64(beta))
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, resampledesc)
        index === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, index)
    end
end

function resample_backward_operation(resampledesc::cudnnBackendDescriptor,
                                     dx::cudnnBackendDescriptor,
                                     dy::cudnnBackendDescriptor;
                                     x::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                     y::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                     index::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                     alpha::Real=1, beta::Real=0)
    make_descriptor(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, dx)
        x === nothing || setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC, x)
        y === nothing || setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC, y)
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, dy)
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA, Float64(alpha))
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA, Float64(beta))
        setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC, resampledesc)
        index === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, index)
    end
end

function norm_forward_operation(; mode::cudnnBackendNormMode_t,
                                phase::cudnnBackendNormFwdPhase_t,
                                x::cudnnBackendDescriptor,
                                mean::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                inv_variance::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                scale::cudnnBackendDescriptor,
                                bias::cudnnBackendDescriptor,
                                epsilon::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                momentum::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                input_running_mean::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                input_running_var::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                output_running_mean::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                output_running_var::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                y::cudnnBackendDescriptor)
    make_descriptor(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, mode)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, phase)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, x)
        mean === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, mean)
        inv_variance === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, inv_variance)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, scale)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, bias)
        epsilon === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, epsilon)
        momentum === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC, momentum)
        input_running_mean === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                     input_running_mean)
        input_running_var === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                     input_running_var)
        output_running_mean === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                     output_running_mean)
        output_running_var === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
                     output_running_var)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, y)
    end
end

function norm_backward_operation(; mode::cudnnBackendNormMode_t,
                                 x::cudnnBackendDescriptor,
                                 mean::cudnnBackendDescriptor,
                                 inv_variance::cudnnBackendDescriptor,
                                 dy::cudnnBackendDescriptor,
                                 scale::cudnnBackendDescriptor,
                                 dscale::cudnnBackendDescriptor,
                                 dbias::cudnnBackendDescriptor,
                                 dx::cudnnBackendDescriptor)
    make_descriptor(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_MODE, mode)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, x)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, mean)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, inv_variance)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, dy)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, scale)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, dscale)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, dbias)
        setattr!(op, CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, dx)
    end
end

function diagonal_band_mask_operation(x::cudnnBackendDescriptor, b::cudnnBackendDescriptor,
                                      y::cudnnBackendDescriptor;
                                      seq_len_q::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                      seq_len_kv::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                      cu_seq_len_q::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                      cu_seq_len_kv::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                      left_bound::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                      shift_right_bound::Union{Nothing,cudnnBackendDescriptor}=nothing,
                                      comparison_mode::cudnnPointwiseMode_t)
    make_descriptor(CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR) do op
        setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC, x)
        setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC, b)
        setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC, y)
        setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE,
                 comparison_mode)
        seq_len_q === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC, seq_len_q)
        seq_len_kv === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC, seq_len_kv)
        cu_seq_len_q === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC,
                     cu_seq_len_q)
        cu_seq_len_kv === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC,
                     cu_seq_len_kv)
        left_bound === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC,
                     left_bound)
        shift_right_bound === nothing ||
            setattr!(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC,
                     shift_right_bound)
    end
end

# Return caller-owned engine-config descriptors the heuristic suggests, in preference order.
function engine_configs(graph::cudnnBackendDescriptor;
                        deviceprop::Union{Nothing,cudnnBackendDescriptor}=nothing,
                        mode::cudnnBackendHeurMode_t=CUDNN_HEUR_MODE_A, maxcount::Integer=16)
    heur = make_descriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR) do heur
        setattr!(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph)
        setattr!(heur, CUDNN_ATTR_ENGINEHEUR_MODE, mode)
        deviceprop !== nothing && setattr!(heur, CUDNN_ATTR_ENGINEHEUR_DEVICEPROP, deviceprop)
    end
    try
        count = min(Int64(maxcount), getattr_count(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR))
        return getattr_descriptors(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                   CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, count)
    finally
        unsafe_destroy!(heur)
    end
end

is_unsupported(e::CUDNNError) = 3000 <= Int(e.code) < 4000

# Build and finalize an execution plan for an engine config. Returns `nothing` if cuDNN
# reports the config as not supported (a normal outcome: callers iterate configs until one
# finalizes). Any other error (e.g. BAD_PARAM, which indicates a graph-construction bug
# rather than an unsupported config) is rethrown.
function try_execution_plan(enginecfg::cudnnBackendDescriptor;
                            deviceprop::Union{Nothing,cudnnBackendDescriptor}=nothing)
    try
        return make_descriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR) do plan
            setattr_handle!(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE)
            setattr!(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, enginecfg)
            deviceprop !== nothing && setattr!(plan, CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP, deviceprop)
        end
    catch e
        e isa CUDNNError || rethrow()
        is_unsupported(e) || rethrow()
        return nothing
    end
end

plan_workspace_size(plan) = getattr_int64(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)

function engine_descriptor(enginecfg::cudnnBackendDescriptor)
    descs = getattr_descriptors(enginecfg, CUDNN_ATTR_ENGINECFG_ENGINE,
                                CUDNN_BACKEND_ENGINE_DESCRIPTOR, 1)
    isempty(descs) && error("cuDNN: engine config did not expose an engine descriptor")
    return only(descs)
end

engine_numerical_notes(engine::cudnnBackendDescriptor) =
    getattr(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE,
            cudnnBackendNumericalNote_t,
            getattr_count(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                          CUDNN_TYPE_NUMERICAL_NOTE))

engine_behavior_notes(engine::cudnnBackendDescriptor) =
    getattr(engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE,
            cudnnBackendBehaviorNote_t,
            getattr_count(engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                          CUDNN_TYPE_BEHAVIOR_NOTE))

# Build and finalize a variant pack. `pointers` are device (or, for by-value tensors, host)
# pointers matching `uids` one-to-one; `workspace` is a device pointer or C_NULL.
function variant_pack(; uids::AbstractVector{<:Integer}, pointers::AbstractVector,
                      workspace)
    make_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR) do vp
        ptrbuf = [reinterpret(Ptr{Cvoid}, p) for p in pointers]
        setattr!(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, uids)
        setattr!(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,
                 length(ptrbuf), ptrbuf)
        setattr!(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
                 Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, workspace)])
    end
end
