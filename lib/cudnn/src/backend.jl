mutable struct BackendDescriptor
    ptr::cudnnBackendDescriptor_t
    descriptor_type::cudnnBackendDescriptorType_t
end

function BackendDescriptor(descriptor_type::cudnnBackendDescriptorType_t)
    ref = Ref{cudnnBackendDescriptor_t}(C_NULL)
    cudnnBackendCreateDescriptor(descriptor_type, ref)
    d = BackendDescriptor(ref[], descriptor_type)
    finalizer(unsafe_destroy!, d)
    return d
end

Base.unsafe_convert(::Type{cudnnBackendDescriptor_t}, d::BackendDescriptor) = d.ptr

function unsafe_destroy!(d::BackendDescriptor)
    ptr = d.ptr
    ptr == C_NULL && return
    d.ptr = C_NULL
    cudnnBackendDestroyDescriptor(ptr)
    return
end

function make_descriptor(f, type::cudnnBackendDescriptorType_t)
    d = BackendDescriptor(type)
    try
        f(d)
        cudnnBackendFinalize(d)
        return d
    catch
        unsafe_destroy!(d)
        rethrow()
    end
end


## Attributes

# Keep attribute storage alive across the C call.
function setattr!(d::BackendDescriptor, name::cudnnBackendAttributeName_t,
                  atype::cudnnBackendAttributeType_t, count::Integer, buf)
    GC.@preserve buf begin
        cudnnBackendSetAttribute(d.ptr, name, atype, Int64(count),
                                 convert(Ptr{Cvoid}, pointer(buf)))
    end
    return d
end

setattr!(d, name, v::Integer) = setattr!(d, name, CUDNN_TYPE_INT64, 1, Int64[v])
setattr!(d, name, v::AbstractVector{<:Integer}) =
    setattr!(d, name, CUDNN_TYPE_INT64, length(v), convert(Vector{Int64}, v))

setattr!(d, name, v::Bool) = setattr!(d, name, CUDNN_TYPE_BOOLEAN, 1, Bool[v])

setattr!(d, name, v::Float32) = setattr!(d, name, CUDNN_TYPE_FLOAT, 1, Float32[v])
setattr!(d, name, v::Float64) = setattr!(d, name, CUDNN_TYPE_DOUBLE, 1, Float64[v])

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

setattr_handle!(d) =
    setattr!(d, attribute(d, :handle), CUDNN_TYPE_HANDLE, 1,
             cudnnHandle_t[Base.unsafe_convert(cudnnHandle_t, handle())])

setattr!(d, name, v::BackendDescriptor) =
    setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, cudnnBackendDescriptor_t[v.ptr])
setattr!(d, name, v::AbstractVector{BackendDescriptor}) =
    setattr!(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, length(v),
             cudnnBackendDescriptor_t[x.ptr for x in v])


## Attribute reads
function getattr(d::BackendDescriptor, name::cudnnBackendAttributeName_t,
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

getattr(d::BackendDescriptor, name::cudnnBackendAttributeName_t, ::Type{T},
        maxcount::Integer=1) where {T} =
    getattr(d, name, backend_attribute_type(T), T, maxcount)

backend_attribute_type(::Type{Int64}) = CUDNN_TYPE_INT64
backend_attribute_type(::Type{Float32}) = CUDNN_TYPE_FLOAT
backend_attribute_type(::Type{Float64}) = CUDNN_TYPE_DOUBLE
backend_attribute_type(::Type{Bool}) = CUDNN_TYPE_BOOLEAN

function getattr_count(d::BackendDescriptor, name::cudnnBackendAttributeName_t,
                       atype::cudnnBackendAttributeType_t)
    n = Ref{Int64}(0)
    cudnnBackendGetAttribute(d.ptr, name, atype, Int64(0), n, C_NULL)
    return n[]
end

# cuDNN requires preallocated output descriptors; destroy unused ones immediately.
function getattr_descriptors(d::BackendDescriptor, name::cudnnBackendAttributeName_t,
                             desctype::cudnnBackendDescriptorType_t, maxcount::Integer)
    maxcount == 0 && return BackendDescriptor[]
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
    nreturned == 0 && return BackendDescriptor[]
    return map(1:nreturned) do i
        desc = BackendDescriptor(raw[i], desctype)
        finalizer(unsafe_destroy!, desc)
        desc
    end
end


## Symbolic attributes

# Map descriptor fields to their CUDNN_ATTR_* names, including cuDNN abbreviations.

function attribute_prefixes(stem::String)
    variants = [stem, replace(stem, "BACKWARD" => "BWD", "FORWARD" => "FWD")]
    stem == "OPERATION_BN_FINALIZE_STATISTICS" && push!(variants, "OPERATION_BN_FINALIZE")
    stem == "OPERATION_GEN_STATS" && push!(variants, "OPERATION_GENSTATS")
    stem == "OPERATION_CONTRACT_BAND_MATRIX" && push!(variants, "OPERATION_CONTRACT_BAND")
    return unique(variants)
end

const descriptor_types, attribute_names = let
    types = Dict{Symbol,cudnnBackendDescriptorType_t}()
    stems = Dict{cudnnBackendDescriptorType_t,Vector{String}}()
    for d in instances(cudnnBackendDescriptorType_t)
        name = string(d)
        endswith(name, "_DESCRIPTOR") || continue
        stem = String(chopsuffix(chopprefix(name, "CUDNN_BACKEND_"), "_DESCRIPTOR"))
        types[Symbol(lowercase(stem))] = d
        stems[d] = attribute_prefixes(stem)
    end
    attributes = Dict{Tuple{cudnnBackendDescriptorType_t,Symbol},cudnnBackendAttributeName_t}()
    for a in instances(cudnnBackendAttributeName_t)
        body = chopprefix(string(a), "CUDNN_ATTR_")
        best, bestlen = nothing, 0
        for (d, prefixes) in stems, p in prefixes
            if startswith(body, p * "_") && length(p) > bestlen
                best, bestlen = (d, p), length(p)
            end
        end
        best === nothing && continue
        d, p = best
        attributes[(d, Symbol(lowercase(chopprefix(body, p * "_"))))] = a
    end
    types, attributes
end

function attribute(d::BackendDescriptor, name::Symbol)
    attr = get(attribute_names, (d.descriptor_type, name), nothing)
    attr === nothing &&
        throw(ArgumentError("$(d.descriptor_type) has no attribute $name; valid attributes are " *
                            join(sort([string(f) for (t, f) in keys(attribute_names)
                                       if t == d.descriptor_type]), ", ")))
    return attr
end

Base.setindex!(d::BackendDescriptor, v, name::Symbol) = setattr!(d, attribute(d, name), v)

function Base.getindex(d::BackendDescriptor, name::Symbol, ::Type{Vector{T}}) where {T}
    attr = attribute(d, name)
    atype = backend_attribute_type(T)
    return getattr(d, attr, atype, T, getattr_count(d, attr, atype))
end
Base.getindex(d::BackendDescriptor, name::Symbol, ::Type{T}) where {T} =
    only(getattr(d, attribute(d, name), backend_attribute_type(T), T, 1))

BackendDescriptor(type::Symbol) = BackendDescriptor(descriptor_types[type])
make_descriptor(f, type::Symbol) = make_descriptor(f, descriptor_types[type])


## Constructors

function backend_tensor(; uid::Integer, dims, strides, dtype::cudnnDataType_t,
                        is_virtual::Bool=false, by_value::Bool=false, alignment::Integer=16)
    make_descriptor(:tensor) do d
        d[:unique_id] = Int64(uid)
        d[:data_type] = dtype
        d[:dimensions] = dims
        d[:strides] = strides
        d[:byte_alignment] = Int64(alignment)
        is_virtual && (d[:is_virtual] = true)
        by_value && (d[:is_by_value] = true)
    end
end

function backend_deviceprop()
    make_descriptor(:deviceprop) do d
        setattr_handle!(d)
    end
end

function operation_graph(ops::AbstractVector{BackendDescriptor};
                         mode::cudnnBackendOperationGraphMode_t=CUDNN_OPERATIONGRAPH_MODE_AUTO)
    make_descriptor(:operationgraph) do g
        setattr_handle!(g)
        g[:ops] = ops
        mode == CUDNN_OPERATIONGRAPH_MODE_AUTO || (g[:mode] = mode)
    end
end

function pointwise_descriptor(; mode::cudnnPointwiseMode_t, compute_type::cudnnDataType_t,
                              nan_propagation::cudnnNanPropagation_t=CUDNN_NOT_PROPAGATE_NAN,
                              relu_lower_clip::Real=0, relu_upper_clip::Real=Inf,
                              relu_lower_clip_slope::Real=0, elu_alpha::Real=1,
                              softplus_beta::Real=1, swish_beta::Real=1,
                              axis::Union{Nothing,Integer}=nothing)
    make_descriptor(:pointwise) do d
        d[:mode] = mode
        d[:math_prec] = compute_type
        if mode in (CUDNN_POINTWISE_RELU_FWD, CUDNN_POINTWISE_RELU_BWD)
            d[:nan_propagation] = nan_propagation
            d[:relu_lower_clip] = Float64(relu_lower_clip)
            d[:relu_upper_clip] = Float64(relu_upper_clip)
            d[:relu_lower_clip_slope] = Float64(relu_lower_clip_slope)
        elseif mode in (CUDNN_POINTWISE_ELU_FWD, CUDNN_POINTWISE_ELU_BWD)
            d[:elu_alpha] = Float64(elu_alpha)
        elseif mode == CUDNN_POINTWISE_SOFTPLUS_FWD
            d[:softplus_beta] = Float64(softplus_beta)
        elseif mode in (CUDNN_POINTWISE_SWISH_FWD, CUDNN_POINTWISE_SWISH_BWD)
            d[:swish_beta] = Float64(swish_beta)
        end
        axis === nothing || (d[:axis] = Int64(axis))
    end
end

function matmul_descriptor(; compute_type::cudnnDataType_t)
    make_descriptor(:matmul) do d
        d[:comp_type] = compute_type
    end
end

function reduction_descriptor(; mode::cudnnReduceTensorOp_t, compute_type::cudnnDataType_t,
                              deterministic::Bool=false)
    make_descriptor(:reduction) do d
        d[:comp_type] = compute_type
        d[:operator] = mode
        d[:is_deterministic] = deterministic
    end
end

function backend_convolution_descriptor(; compute_type::cudnnDataType_t,
                                        mode::cudnnConvolutionMode_t,
                                        pre_padding, post_padding, dilation, stride)
    spatial_dims = length(pre_padding)
    length(post_padding) == spatial_dims && length(dilation) == spatial_dims &&
        length(stride) == spatial_dims ||
        throw(DimensionMismatch("convolution attributes must have matching rank"))
    make_descriptor(:convolution) do d
        d[:comp_type] = compute_type
        d[:conv_mode] = mode
        d[:spatial_dims] = spatial_dims
        d[:pre_paddings] = pre_padding
        d[:post_paddings] = post_padding
        d[:dilations] = dilation
        d[:filter_strides] = stride
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
    make_descriptor(:resample) do d
        d[:mode] = mode
        d[:comp_type] = compute_type
        d[:nan_propagation] = nan_propagation
        d[:padding_mode] = padding_mode
        d[:spatial_dims] = spatial_dims
        d[:window_dims] = fractions(window)
        d[:pre_paddings] = fractions(pre_padding)
        d[:post_paddings] = fractions(post_padding)
        d[:strides] = fractions(stride)
    end
end

function pointwise_operation(pwdesc::BackendDescriptor, x::BackendDescriptor,
                             y::BackendDescriptor;
                             b::Union{Nothing,BackendDescriptor}=nothing,
                             t::Union{Nothing,BackendDescriptor}=nothing,
                             alpha1::Real=1, alpha2::Real=0)
    make_descriptor(:operation_pointwise) do op
        op[:pw_descriptor] = pwdesc
        op[:xdesc] = x
        op[:ydesc] = y
        op[:alpha1] = Float32(alpha1)
        op[:alpha2] = Float32(alpha2)
        b === nothing || (op[:bdesc] = b)
        t === nothing || (op[:tdesc] = t)
    end
end

function matmul_operation(matdesc::BackendDescriptor, a::BackendDescriptor,
                          b::BackendDescriptor, c::BackendDescriptor)
    make_descriptor(:operation_matmul) do op
        op[:adesc] = a
        op[:bdesc] = b
        op[:cdesc] = c
        op[:desc] = matdesc
    end
end

function reduction_operation(reddesc::BackendDescriptor, x::BackendDescriptor,
                             y::BackendDescriptor)
    make_descriptor(:operation_reduction) do op
        op[:desc] = reddesc
        op[:xdesc] = x
        op[:ydesc] = y
    end
end

function convolution_forward_operation(convdesc::BackendDescriptor,
                                       x::BackendDescriptor,
                                       w::BackendDescriptor,
                                       y::BackendDescriptor;
                                       alpha::Real=1, beta::Real=0,
                                       alphabeta_type::cudnnBackendAttributeType_t=CUDNN_TYPE_FLOAT)
    make_descriptor(:operation_convolution_forward) do op
        op[:conv_desc] = convdesc
        op[:x] = x
        op[:w] = w
        op[:y] = y
        setattr_alphabeta!(op, :alpha, alpha, alphabeta_type)
        setattr_alphabeta!(op, :beta, beta, alphabeta_type)
    end
end

function convolution_data_backward_operation(convdesc::BackendDescriptor,
                                             w::BackendDescriptor,
                                             dy::BackendDescriptor,
                                             dx::BackendDescriptor;
                                             alpha::Real=1, beta::Real=0,
                                             alphabeta_type::cudnnBackendAttributeType_t=CUDNN_TYPE_FLOAT)
    make_descriptor(:operation_convolution_backward_data) do op
        op[:conv_desc] = convdesc
        op[:w] = w
        op[:dy] = dy
        op[:dx] = dx
        setattr_alphabeta!(op, :alpha, alpha, alphabeta_type)
        setattr_alphabeta!(op, :beta, beta, alphabeta_type)
    end
end

function convolution_filter_backward_operation(convdesc::BackendDescriptor,
                                               x::BackendDescriptor,
                                               dy::BackendDescriptor,
                                               dw::BackendDescriptor;
                                               alpha::Real=1, beta::Real=0,
                                               alphabeta_type::cudnnBackendAttributeType_t=CUDNN_TYPE_FLOAT)
    make_descriptor(:operation_convolution_backward_filter) do op
        op[:conv_desc] = convdesc
        op[:x] = x
        op[:dy] = dy
        op[:dw] = dw
        setattr_alphabeta!(op, :alpha, alpha, alphabeta_type)
        setattr_alphabeta!(op, :beta, beta, alphabeta_type)
    end
end

# alpha and beta use the convolution compute type.
function setattr_alphabeta!(d::BackendDescriptor, name::Symbol, value, atype)
    if atype == CUDNN_TYPE_DOUBLE
        setattr!(d, attribute(d, name), CUDNN_TYPE_DOUBLE, 1, Float64[value])
    else
        setattr!(d, attribute(d, name), CUDNN_TYPE_FLOAT, 1, Float32[value])
    end
end

function resample_forward_operation(resampledesc::BackendDescriptor,
                                    x::BackendDescriptor,
                                    y::BackendDescriptor;
                                    index::Union{Nothing,BackendDescriptor}=nothing,
                                    alpha::Real=1, beta::Real=0)
    make_descriptor(:operation_resample_fwd) do op
        op[:xdesc] = x
        op[:ydesc] = y
        op[:alpha] = Float64(alpha)
        op[:beta] = Float64(beta)
        op[:desc] = resampledesc
        index === nothing || (op[:idxdesc] = index)
    end
end

function resample_backward_operation(resampledesc::BackendDescriptor,
                                     dx::BackendDescriptor,
                                     dy::BackendDescriptor;
                                     x::Union{Nothing,BackendDescriptor}=nothing,
                                     y::Union{Nothing,BackendDescriptor}=nothing,
                                     index::Union{Nothing,BackendDescriptor}=nothing,
                                     alpha::Real=1, beta::Real=0)
    make_descriptor(:operation_resample_bwd) do op
        op[:dxdesc] = dx
        x === nothing || (op[:xdesc] = x)
        y === nothing || (op[:ydesc] = y)
        op[:dydesc] = dy
        op[:alpha] = Float64(alpha)
        op[:beta] = Float64(beta)
        op[:desc] = resampledesc
        index === nothing || (op[:idxdesc] = index)
    end
end

function norm_forward_operation(; mode::cudnnBackendNormMode_t,
                                phase::cudnnBackendNormFwdPhase_t,
                                x::BackendDescriptor,
                                mean::Union{Nothing,BackendDescriptor}=nothing,
                                inv_variance::Union{Nothing,BackendDescriptor}=nothing,
                                scale::BackendDescriptor,
                                bias::BackendDescriptor,
                                epsilon::Union{Nothing,BackendDescriptor}=nothing,
                                momentum::Union{Nothing,BackendDescriptor}=nothing,
                                input_running_mean::Union{Nothing,BackendDescriptor}=nothing,
                                input_running_var::Union{Nothing,BackendDescriptor}=nothing,
                                output_running_mean::Union{Nothing,BackendDescriptor}=nothing,
                                output_running_var::Union{Nothing,BackendDescriptor}=nothing,
                                y::BackendDescriptor)
    make_descriptor(:operation_norm_forward) do op
        op[:mode] = mode
        op[:phase] = phase
        op[:xdesc] = x
        mean === nothing || (op[:mean_desc] = mean)
        inv_variance === nothing || (op[:inv_variance_desc] = inv_variance)
        op[:scale_desc] = scale
        op[:bias_desc] = bias
        epsilon === nothing || (op[:epsilon_desc] = epsilon)
        momentum === nothing || (op[:exp_avg_factor_desc] = momentum)
        input_running_mean === nothing || (op[:input_running_mean_desc] = input_running_mean)
        input_running_var === nothing || (op[:input_running_var_desc] = input_running_var)
        output_running_mean === nothing || (op[:output_running_mean_desc] = output_running_mean)
        output_running_var === nothing || (op[:output_running_var_desc] = output_running_var)
        op[:ydesc] = y
    end
end

function norm_backward_operation(; mode::cudnnBackendNormMode_t,
                                 x::BackendDescriptor,
                                 mean::BackendDescriptor,
                                 inv_variance::BackendDescriptor,
                                 dy::BackendDescriptor,
                                 scale::BackendDescriptor,
                                 dscale::BackendDescriptor,
                                 dbias::BackendDescriptor,
                                 dx::BackendDescriptor)
    make_descriptor(:operation_norm_backward) do op
        op[:mode] = mode
        op[:xdesc] = x
        op[:mean_desc] = mean
        op[:inv_variance_desc] = inv_variance
        op[:dydesc] = dy
        op[:scale_desc] = scale
        op[:dscale_desc] = dscale
        op[:dbias_desc] = dbias
        op[:dxdesc] = dx
    end
end

function diagonal_band_mask_operation(x::BackendDescriptor, b::BackendDescriptor,
                                      y::BackendDescriptor;
                                      seq_len_q::Union{Nothing,BackendDescriptor}=nothing,
                                      seq_len_kv::Union{Nothing,BackendDescriptor}=nothing,
                                      cu_seq_len_q::Union{Nothing,BackendDescriptor}=nothing,
                                      cu_seq_len_kv::Union{Nothing,BackendDescriptor}=nothing,
                                      left_bound::Union{Nothing,BackendDescriptor}=nothing,
                                      shift_right_bound::Union{Nothing,BackendDescriptor}=nothing,
                                      comparison_mode::cudnnPointwiseMode_t)
    make_descriptor(:operation_diagonal_band_mask) do op
        op[:xdesc] = x
        op[:bdesc] = b
        op[:ydesc] = y
        op[:comparison_mode] = comparison_mode
        seq_len_q === nothing || (op[:seq_len_qdesc] = seq_len_q)
        seq_len_kv === nothing || (op[:seq_len_kvdesc] = seq_len_kv)
        cu_seq_len_q === nothing || (op[:cu_seq_len_qdesc] = cu_seq_len_q)
        cu_seq_len_kv === nothing || (op[:cu_seq_len_kvdesc] = cu_seq_len_kv)
        left_bound === nothing || (op[:left_bound_desc] = left_bound)
        shift_right_bound === nothing || (op[:shift_right_bound_desc] = shift_right_bound)
    end
end

function engine_configs(graph::BackendDescriptor;
                        deviceprop::Union{Nothing,BackendDescriptor}=nothing,
                        mode::cudnnBackendHeurMode_t=CUDNN_HEUR_MODE_A)
    heur = make_descriptor(:engineheur) do heur
        heur[:operation_graph] = graph
        heur[:mode] = mode
        deviceprop === nothing || (heur[:deviceprop] = deviceprop)
    end
    try
        count = getattr_count(heur, attribute(heur, :results),
                              CUDNN_TYPE_BACKEND_DESCRIPTOR)
        return getattr_descriptors(heur, attribute(heur, :results),
                                   CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, count)
    finally
        unsafe_destroy!(heur)
    end
end

is_unsupported(e::CUDNNError) = 3000 <= Int(e.code) < 4000

# Unsupported configurations are expected while searching for a plan.
function try_execution_plan(enginecfg::BackendDescriptor;
                            deviceprop::Union{Nothing,BackendDescriptor}=nothing)
    try
        return make_descriptor(:execution_plan) do plan
            setattr_handle!(plan)
            plan[:engine_config] = enginecfg
            deviceprop === nothing || (plan[:deviceprop] = deviceprop)
        end
    catch e
        e isa CUDNNError || rethrow()
        is_unsupported(e) || rethrow()
        return nothing
    end
end

plan_workspace_size(plan) = plan[:workspace_size, Int64]

function engine_descriptor(enginecfg::BackendDescriptor)
    descs = getattr_descriptors(enginecfg, attribute(enginecfg, :engine),
                                CUDNN_BACKEND_ENGINE_DESCRIPTOR, 1)
    isempty(descs) && error("cuDNN: engine config did not expose an engine descriptor")
    return only(descs)
end

engine_numerical_notes(engine::BackendDescriptor) =
    engine[:numerical_note, Vector{cudnnBackendNumericalNote_t}]

engine_behavior_notes(engine::BackendDescriptor) =
    engine[:behavior_note, Vector{cudnnBackendBehaviorNote_t}]

function variant_pack(; uids::AbstractVector{<:Integer}, pointers::AbstractVector,
                      workspace)
    make_descriptor(:variant_pack) do vp
        vp[:unique_ids] = uids
        vp[:data_pointers] = Ptr{Cvoid}[reinterpret(Ptr{Cvoid}, p) for p in pointers]
        vp[:workspace] = reinterpret(Ptr{Cvoid}, workspace)
    end
end
