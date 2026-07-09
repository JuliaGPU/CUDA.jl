conv_pointwise_activation(::Nothing) = nothing
conv_pointwise_activation(mode::cudnnPointwiseMode_t) =
    mode == CUDNN_POINTWISE_IDENTITY ? nothing : mode
conv_pointwise_activation(mode::cudnnActivationMode_t) =
    mode == CUDNN_ACTIVATION_IDENTITY ? nothing :
    mode == CUDNN_ACTIVATION_RELU ? CUDNN_POINTWISE_RELU_FWD :
    mode == CUDNN_ACTIVATION_TANH ? CUDNN_POINTWISE_TANH_FWD :
    mode == CUDNN_ACTIVATION_SIGMOID ? CUDNN_POINTWISE_SIGMOID_FWD :
    mode == CUDNN_ACTIVATION_ELU ? CUDNN_POINTWISE_ELU_FWD :
    throw(ArgumentError("unknown convolution activation $mode"))
conv_pointwise_activation(mode::Symbol) =
    mode === :identity ? nothing :
    mode === :relu ? CUDNN_POINTWISE_RELU_FWD :
    mode === :tanh ? CUDNN_POINTWISE_TANH_FWD :
    mode === :sigmoid ? CUDNN_POINTWISE_SIGMOID_FWD :
    mode === :elu ? CUDNN_POINTWISE_ELU_FWD :
    throw(ArgumentError("unknown convolution activation $mode"))

conv_compute_type(::Type{Float16}) = Float32
conv_compute_type(::Type{BFloat16}) = Float32
conv_compute_type(::Type{T}) where {T} = T

function symmetric_padding(padding, spatial_rank)
    padding isa Integer && return padding
    p = collect(Int, padding)
    length(p) == spatial_rank && return p
    if length(p) == 2spatial_rank
        all(p[1:2:end] .== p[2:2:end]) ||
            throw(ArgumentError("legacy convolution path requires symmetric padding"))
        return p[1:2:end]
    end
    throw(DimensionMismatch("padding must have length $spatial_rank or $(2spatial_rank)"))
end

function convolution_descriptor(x, padding, stride, dilation, mode, groups, compute_type)
    cudnnConvolutionDescriptor(convdims(padding, size(x), 0),
                               convdims(stride, size(x), 1),
                               convdims(dilation, size(x), 1),
                               graph_conv_mode(mode), cudnnDataType(compute_type),
                               math_mode(), CUDNN_DEFAULT_REORDER, Cint(groups))
end

function convolution_padding(padding, spatial_rank)
    padding isa Integer && return fill(Int(padding), spatial_rank),
                                  fill(Int(padding), spatial_rank)
    p = collect(Int, padding)
    length(p) == spatial_rank && return p, copy(p)
    if length(p) == 2spatial_rank
        return p[1:2:end], p[2:2:end]
    end
    throw(DimensionMismatch("padding must have length $spatial_rank or $(2spatial_rank)"))
end

# Convert the integer, tuple or array to padding/stride/dilation dims compatible with array size
function convdims(d, s::Dims{N}, default) where N
    @assert d isa Integer || length(d) == N-2  "Cannot use $d padding/stride/dilation with $(Base.dims2string(s)) array."
    if N >= 4
        (d isa Integer ? fill(Cint(d), N-2) : Cint[reverse(d)...])
    elseif N == 3
        Cint[d[1], default]    # 3D tensors are padded to 4D
    else
        Cint[default, default] # lower dim tensors have no spatial dims
    end
end

# symmetric-padding geometry equivalent to convolving x with asymmetric (pre, post)
# padding: the excess padding is materialized as zeros, with x at `ranges` in the
# padded array
function padded_convolution_geometry(x, pre, post)
    rank = length(pre)
    pad = min.(pre, post)
    padded_size = ntuple(ndims(x)) do i
        i <= rank ? size(x, i) + abs(pre[i] - post[i]) : size(x, i)
    end
    ranges = ntuple(ndims(x)) do i
        if i <= rank
            offset = max(pre[i] - post[i], 0)
            (1 + offset):(size(x, i) + offset)
        else
            axes(x, i)
        end
    end
    return pad, padded_size, ranges
end

function padded_convolution_input(x, pre, post)
    pad, padded_size, ranges = padded_convolution_geometry(x, pre, post)
    padded = fill!(similar(x, padded_size), zero(eltype(x)))
    copyto!(view(padded, ranges...), x)
    return padded, pad
end

function convolution_key(y, x, w, pre_padding, post_padding, stride, dilation, mode,
                          compute_type, alpha, beta, deterministic, math_mode, max_workspace)
    (:conv_fprop, eltype(x),
     size(x), strides(x), pointer_alignment(x),
     size(w), strides(w), pointer_alignment(w),
     size(y), strides(y), pointer_alignment(y),
     Tuple(pre_padding), Tuple(post_padding), Tuple(stride), Tuple(dilation), mode,
     compute_type, Float64(alpha), Float64(beta), deterministic, math_mode, max_workspace)
end

conv_optional_array_key(::Nothing) = nothing
conv_optional_array_key(a::DenseCuArray) =
    (size(a), strides(a), pointer_alignment(a))

function fused_convolution_key(y, x, w, z, bias, activation, pre_padding, post_padding,
                                stride, dilation, mode, compute_type, alpha, beta,
                                deterministic, math_mode, max_workspace)
    (:conv_fprop_fused, eltype(x),
     size(x), strides(x), pointer_alignment(x),
     size(w), strides(w), pointer_alignment(w),
     size(y), strides(y), pointer_alignment(y),
     conv_optional_array_key(z), conv_optional_array_key(bias), activation,
     Tuple(pre_padding), Tuple(post_padding), Tuple(stride), Tuple(dilation), mode,
     compute_type, Float64(alpha), Float64(beta), deterministic, math_mode, max_workspace)
end

function convolution_dgrad_key(dx, dy, w, pre_padding, post_padding, stride, dilation,
                                mode, compute_type, alpha, beta, deterministic, math_mode,
                                max_workspace)
    (:conv_dgrad, eltype(dx),
     size(dx), strides(dx), pointer_alignment(dx),
     size(dy), strides(dy), pointer_alignment(dy),
     size(w), strides(w), pointer_alignment(w),
     Tuple(pre_padding), Tuple(post_padding), Tuple(stride), Tuple(dilation), mode,
     compute_type, Float64(alpha), Float64(beta), deterministic, math_mode, max_workspace)
end

function convolution_wgrad_key(dw, x, dy, pre_padding, post_padding, stride, dilation,
                                mode, compute_type, alpha, beta, deterministic, math_mode,
                                max_workspace)
    (:conv_wgrad, eltype(dw),
     size(dw), strides(dw), pointer_alignment(dw),
     size(x), strides(x), pointer_alignment(x),
     size(dy), strides(dy), pointer_alignment(dy),
     Tuple(pre_padding), Tuple(post_padding), Tuple(stride), Tuple(dilation), mode,
     compute_type, Float64(alpha), Float64(beta), deterministic, math_mode, max_workspace)
end

function build_convolution_graph(y, x, w, pre_padding, post_padding, stride, dilation,
                                  mode, compute_type, alpha, beta;
                                  deterministic, math_mode, max_workspace)
    g = Graph(io_dtype=eltype(x), intermediate_dtype=Float32, compute_dtype=compute_type)
    tx = tensor!(g, x; name="X")
    tw = tensor!(g, w; name="W")
    ty = tensor!(g, y; name="Y", output=true)
    conv_fprop!(g, tx, tw; y=ty, pre_padding, post_padding, stride, dilation, mode,
                compute_dtype=compute_type, alpha, beta)
    build!(g; deterministic, math_mode, max_workspace)
end

function check_conv_broadcast(name, a::DenseCuArray, y::DenseCuArray)
    ndims(a) == ndims(y) ||
        throw(DimensionMismatch("$name must have rank $(ndims(y)), got $(ndims(a))"))
    all(i -> size(a, i) == 1 || size(a, i) == size(y, i), 1:ndims(y)) ||
        throw(DimensionMismatch("$name is not broadcastable to output size $(size(y))"))
    eltype(a) == eltype(y) ||
        throw(ArgumentError("$name eltype must be $(eltype(y)), got $(eltype(a))"))
    return a
end

function apply_conv_activation!(y, ::Nothing)
    return y
end
function apply_conv_activation!(y, mode::cudnnPointwiseMode_t)
    if mode == CUDNN_POINTWISE_RELU_FWD
        @. y = max(y, zero(y))
    elseif mode == CUDNN_POINTWISE_TANH_FWD
        @. y = tanh(y)
    elseif mode == CUDNN_POINTWISE_SIGMOID_FWD
        @. y = inv(one(y) + exp(-y))
    elseif mode == CUDNN_POINTWISE_ELU_FWD
        @. y = ifelse(y > zero(y), y, exp(y) - one(y))
    elseif mode == CUDNN_POINTWISE_IDENTITY
        return y
    else
        throw(ArgumentError("unsupported convolution activation $mode"))
    end
    return y
end

function generic_fused_convolution!(y, x, w, z, bias, activation; padding, stride,
                                     dilation, group, mode, alpha, beta, compute_type,
                                     deterministic, math_mode, max_workspace)
    zsrc = z !== nothing && z === y ? copy(y) : z
    convolution!(y, x, w; padding, stride, dilation, group, mode, alpha, beta=0,
                 compute_type, deterministic, math_mode, max_workspace)
    zsrc === nothing || (@. y = y + beta * zsrc)
    bias === nothing || (@. y = y + bias)
    apply_conv_activation!(y, activation)
end

function build_fused_convolution_graph(y, x, w, z, bias, activation, pre_padding,
                                        post_padding, stride, dilation, mode,
                                        compute_type, alpha, beta;
                                        deterministic, math_mode, max_workspace)
    g = Graph(io_dtype=eltype(x), intermediate_dtype=Float32, compute_dtype=compute_type)
    tx = tensor!(g, x; name="X")
    tw = tensor!(g, w; name="W")
    ty = tensor!(g, y; name="Y", output=true)
    tz = z === nothing ? nothing : tensor!(g, z; name="Z")
    tbias = bias === nothing ? nothing : tensor!(g, bias; name="Bias")
    tconv = tensor!(g; dims=size(y), dtype=compute_type, virtual=true, name="Conv",
                    backend_order=ty.backend_order)
    conv_fprop!(g, tx, tw; y=tconv, pre_padding, post_padding, stride, dilation, mode,
                compute_dtype=compute_type, alpha, beta=0)

    t = tconv
    if tz !== nothing
        out = tbias === nothing && activation === nothing ? ty : nothing
        t = pointwise!(g, :add, t, tz; y=out, compute_dtype=compute_type, alpha1=1,
                       alpha2=beta, name="ZAdd")
    end
    if tbias !== nothing
        out = activation === nothing ? ty : nothing
        t = pointwise!(g, :add, t, tbias; y=out, compute_dtype=compute_type,
                       name="BiasAdd")
    end
    activation === nothing ||
        pointwise!(g, activation, t; y=ty, compute_dtype=compute_type, name="Activation")
    build!(g; deterministic, math_mode, max_workspace)
end

function build_convolution_dgrad_graph(dx, dy, w, pre_padding, post_padding, stride,
                                        dilation, mode, compute_type, alpha, beta;
                                        deterministic, math_mode, max_workspace)
    g = Graph(io_dtype=eltype(dx), intermediate_dtype=Float32, compute_dtype=compute_type)
    tdy = tensor!(g, dy; name="dY")
    tw = tensor!(g, w; name="W")
    tdx = tensor!(g, dx; name="dX", output=true)
    conv_dgrad!(g, tdy, tw; dx=tdx, pre_padding, post_padding, stride, dilation, mode,
                compute_dtype=compute_type, alpha, beta)
    build!(g; deterministic, math_mode, max_workspace)
end

function build_convolution_wgrad_graph(dw, x, dy, pre_padding, post_padding, stride,
                                        dilation, mode, compute_type, alpha, beta;
                                        deterministic, math_mode, max_workspace)
    g = Graph(io_dtype=eltype(dw), intermediate_dtype=Float32, compute_dtype=compute_type)
    tx = tensor!(g, x; name="X")
    tdy = tensor!(g, dy; name="dY")
    tdw = tensor!(g, dw; name="dW", output=true)
    conv_wgrad!(g, tdy, tx; dw=tdw, pre_padding, post_padding, stride, dilation, mode,
                compute_dtype=compute_type, alpha, beta)
    build!(g; deterministic, math_mode, max_workspace)
end

## Utilities to find a fast algorithm

const cudnnConvolutionFwdAlgoPerfCache = Dict{Tuple,cudnnConvolutionFwdAlgoPerf_t}()
const cudnnConvolutionFwdAlgoPerfCacheLock = ReentrantLock()

"""
    cudnnConvolutionFwdAlgoPerf(xDesc, x, wDesc, w, convDesc, yDesc, y, biasDesc, activation, allocateTmpBuf=true)

`allocateTmpBuf` controls whether a temporary buffer is allocated for the output y.
It can be set to false when beta is zero to save an allocation and must otherwise be set to true.
"""
function cudnnConvolutionFwdAlgoPerf(xDesc, x, wDesc, w, convDesc, yDesc, y, biasDesc, activation, allocateTmpBuf=true)
    xDesc_native = cudnnGetTensorDescriptor(xDesc)
    wDesc_native = cudnnGetFilterDescriptor(wDesc)
    convDesc_native = cudnnGetConvolutionDescriptor(convDesc)
    biasDesc_native = (isnothing(biasDesc) ? nothing
                                           : cudnnGetTensorDescriptor(biasDesc))

    key = (xDesc_native, wDesc_native, convDesc_native, biasDesc, activation)
    val = lock(cudnnConvolutionFwdAlgoPerfCacheLock) do
        get(cudnnConvolutionFwdAlgoPerfCache, key, nothing)
    end
    if val === nothing
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionFwdAlgoPerf_t}(undef,requestedAlgoCount)
        workspaceSize() = cudnnFindConvolutionAlgorithmWorkspaceSize(x)
        yTmp = allocateTmpBuf ? similar(y) : y  # if beta is zero we can avoid the allocation
        with_workspace(workspaceSize) do workspace
            cudnnFindConvolutionForwardAlgorithmEx(handle(),xDesc,x,wDesc,w,convDesc,yDesc,yTmp,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,sizeof(workspace))
        end
        val = cudnnConvolutionAlgoPerfChoose(convDesc, xDesc, perfResults, returnedAlgoCount[1])
        lock(cudnnConvolutionFwdAlgoPerfCacheLock) do
            cudnnConvolutionFwdAlgoPerfCache[key] = val
        end
    end
    return val
end

const cudnnConvolutionBwdDataAlgoPerfCache = Dict{Tuple,cudnnConvolutionBwdDataAlgoPerf_t}()
const cudnnConvolutionBwdDataAlgoPerfCacheLock = ReentrantLock()

"""
    cudnnConvolutionBwdDataAlgoPerf(wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, allocateTmpBuf=true)

`allocateTmpBuf` controls whether a temporary buffer is allocated for the input gradient dx.
It can be set to false when beta is zero to save an allocation and must otherwise be set to true.
"""
function cudnnConvolutionBwdDataAlgoPerf(wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, allocateTmpBuf=true)
    wDesc_native = cudnnGetFilterDescriptor(wDesc)
    dyDesc_native = cudnnGetTensorDescriptor(dyDesc)
    convDesc_native = cudnnGetConvolutionDescriptor(convDesc)

    key = (wDesc_native, dyDesc_native, convDesc_native)
    val = lock(cudnnConvolutionBwdDataAlgoPerfCacheLock) do
        get(cudnnConvolutionBwdDataAlgoPerfCache, key, nothing)
    end
    if val === nothing
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionBwdDataAlgoPerf_t}(undef,requestedAlgoCount)
        workspaceSize() = cudnnFindConvolutionAlgorithmWorkspaceSize(dx)
        dxTmp = allocateTmpBuf ? similar(dx) : dx
        with_workspace(workspaceSize) do workspace
            cudnnFindConvolutionBackwardDataAlgorithmEx(handle(),wDesc,w,dyDesc,dy,convDesc,dxDesc,dxTmp,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,sizeof(workspace))
        end
        val = cudnnConvolutionAlgoPerfChoose(convDesc, dyDesc, perfResults, returnedAlgoCount[1])
        lock(cudnnConvolutionBwdDataAlgoPerfCacheLock) do
            cudnnConvolutionBwdDataAlgoPerfCache[key] = val
        end
    end
    val
end

const cudnnConvolutionBwdFilterAlgoPerfCache = Dict{Tuple,cudnnConvolutionBwdFilterAlgoPerf_t}()
const cudnnConvolutionBwdFilterAlgoPerfCacheLock = ReentrantLock()

"""
    cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, dyDesc, dy, convDesc, dwDesc, dw, allocateTmpBuf=true)

`allocateTmpBuf` controls whether a temporary buffer is allocated for the weight gradient dw.
It can be set to false when beta is zero to save an allocation and must otherwise be set to true.
"""
function cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, dyDesc, dy, convDesc, dwDesc, dw, allocateTmpBuf=true)
    xDesc_native = cudnnGetTensorDescriptor(xDesc)
    dyDesc_native = cudnnGetTensorDescriptor(dyDesc)
    convDesc_native = cudnnGetConvolutionDescriptor(convDesc)

    key = (xDesc_native, dyDesc_native, convDesc_native)
    val = lock(cudnnConvolutionBwdFilterAlgoPerfCacheLock) do
        get(cudnnConvolutionBwdFilterAlgoPerfCache, key, nothing)
    end
    if val === nothing
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionBwdFilterAlgoPerf_t}(undef,requestedAlgoCount)
        workspaceSize() = cudnnFindConvolutionAlgorithmWorkspaceSize(x)
        dwTmp = allocateTmpBuf ? similar(dw) : dw
        with_workspace(workspaceSize) do workspace
            cudnnFindConvolutionBackwardFilterAlgorithmEx(handle(),xDesc,x,dyDesc,dy,convDesc,dwDesc,dwTmp,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,sizeof(workspace))
        end
        val = cudnnConvolutionAlgoPerfChoose(convDesc, xDesc, perfResults, returnedAlgoCount[1])
        lock(cudnnConvolutionBwdFilterAlgoPerfCacheLock) do
            cudnnConvolutionBwdFilterAlgoPerfCache[key] = val
        end
    end
    val
end


# Return algorithm with best memory that is within 10% of best time
function cudnnConvolutionAlgoPerfChoose(convDesc, tensorDesc, perfResults, n)
    mathType = Ref{cudnnMathType_t}(CUDNN_DEFAULT_MATH)
    cudnnGetConvolutionMathType(convDesc, mathType)
    skipMathTypeCheck = let  # See https://github.com/JuliaGPU/CUDA.jl/pull/1943#issuecomment-1605267932
        dtype, _ = cudnnGetTensorDescriptor(tensorDesc)
        dtype == Float32
    end

    ibest, mbest, tbest = 0, Inf, Inf
    for (i, ps) in enumerate(perfResults)
        i > n && break
        # These metrics are written in a sorted fashion where the first element has the lowest compute time.
        if ((skipMathTypeCheck || ps.mathType == mathType[])
            && ps.status == CUDNN_STATUS_SUCCESS && ps.memory < mbest && ps.time < tbest * 1.1)
            ibest, mbest, tbest = i, ps.memory, ps.time
        end
    end
    if ibest == 0
        @warn "No valid algorithm found, probably bad params for convolution." maxlog=1
        ibest = findfirst(p->p.algo==0, perfResults)
        ibest === nothing && error("Cannot find backup algorithm for convolution, giving up.")
    end
    return perfResults[ibest]
end


# Allocate the maximum reasonable amount of memory for algorithm discovery
function cudnnFindConvolutionAlgorithmWorkspaceSize(x)
    # Because algorithm discovery runs infrequently yet allocates more than conv functions,
    # This is a good place to synchronize and trim the memory pool to reduce fragmentation.
    CUDACore.reclaim()
    gpufree = CUDACore.free_memory() + coalesce(CUDACore.cached_memory(), 0)
    min(gpufree ÷ 10, sizeof(x) * 100)
end


function legacy_convolution!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T},
                              padding, stride, dilation, group, mode, alpha, beta,
                              compute_type) where {T}
    desc = convolution_descriptor(x, padding, stride, dilation, mode, group, compute_type)
    alpha, beta = scalingParameter(T, alpha), scalingParameter(T, beta)
    xdesc, ydesc, wdesc = cudnnTensorDescriptor(x), cudnnTensorDescriptor(y), cudnnFilterDescriptor(w)
    perf = cudnnConvolutionFwdAlgoPerf(xdesc, x, wdesc, w, desc, ydesc, y, nothing,
                                       CUDNN_ACTIVATION_IDENTITY, beta[] != 0)
    with_workspace(perf.memory) do workspace
        cudnnConvolutionForward(handle(), alpha, xdesc, x, wdesc, w, desc, perf.algo,
                                workspace, sizeof(workspace), beta, ydesc, y)
    end
    return y
end

function convolution!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T};
                      padding=0, stride=1, dilation=1, groups::Integer=1,
                      group::Integer=groups, mode=:cross_correlation,
                      alpha::Real=1, beta::Real=0, bias=nothing,
                      activation=nothing, z=nothing, compute_type=nothing,
                      deterministic::Bool=false, math_mode=CUDACore.math_mode(),
                      max_workspace::Union{Nothing,Integer}=nothing) where {T}
    isempty(y) && return y
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    dil = spatial_vector(dilation, spatial_rank)
    ctype = something(compute_type, conv_compute_type(T))
    cmode = graph_conv_mode(mode)
    pact = conv_pointwise_activation(activation)
    zactive = z !== nothing && beta != 0
    size(x, ndims(x)-1) == size(w, ndims(w)-1) * group ||
        throw(DimensionMismatch("input channels must equal filter channels times group"))
    if bias === nothing && !zactive && pact === nothing
        key = convolution_key(y, x, w, pre, post, str, dil, cmode, ctype, alpha, beta,
                               deterministic, math_mode, max_workspace)
        try
            g = cached_graph(key) do
                build_convolution_graph(y, x, w, pre, post, str, dil, cmode, ctype,
                                         alpha, beta; deterministic, math_mode,
                                         max_workspace)
            end
            execute!(g, tensor(g, "X")=>x, tensor(g, "W")=>w, tensor(g, "Y")=>y)
            return y
        catch e
            graph_unsupported(e) || rethrow()
        end
        if pre != post
            # no engine for asymmetric padding: pad the input manually and retry
            padded_x, pad = padded_convolution_input(x, pre, post)
            return convolution!(y, padded_x, w; padding=pad, stride=str, dilation=dil,
                                group, mode, alpha, beta, compute_type=ctype,
                                deterministic, math_mode, max_workspace)
        end
    else
        zarg = zactive ? check_conv_broadcast("z", z, y) : nothing
        biasarg = bias === nothing ? nothing : check_conv_broadcast("bias", bias, y)
        key = fused_convolution_key(y, x, w, zarg, biasarg, pact, pre, post, str, dil,
                                     cmode, ctype, alpha, beta, deterministic, math_mode,
                                     max_workspace)
        try
            g = cached_graph(key) do
                build_fused_convolution_graph(y, x, w, zarg, biasarg, pact, pre, post,
                                               str, dil, cmode, ctype, alpha, beta;
                                               deterministic, math_mode, max_workspace)
            end
            bindings = IdDict{Tensor,Any}(
                tensor(g, "X") => x,
                tensor(g, "W") => w,
                tensor(g, "Y") => y,
            )
            zarg === nothing || (bindings[tensor(g, "Z")] = zarg)
            biasarg === nothing || (bindings[tensor(g, "Bias")] = biasarg)
            execute!(g, bindings)
            return y
        catch e
            graph_unsupported(e) || rethrow()
            return generic_fused_convolution!(y, x, w, zarg, biasarg, pact; padding,
                                               stride, dilation, group, mode, alpha, beta,
                                               compute_type=ctype, deterministic,
                                               math_mode, max_workspace)
        end
    end
    # only the plain path falls through: bias/z/activation cases returned above
    pad = symmetric_padding(padding, spatial_rank)
    legacy_convolution!(y, x, w, pad, str, dil, group, mode, alpha, beta, ctype)
end

function convolution_data_gradient!(dx::DenseCuArray{T}, dy::DenseCuArray{T},
                                    w::DenseCuArray{T};
                                    padding=0, stride=1, dilation=1,
                                    groups::Integer=1, group::Integer=groups,
                                    mode=:cross_correlation, alpha::Real=1,
                                    beta::Real=0, compute_type=nothing,
                                    deterministic::Bool=false,
                                    math_mode=CUDACore.math_mode(),
                                    max_workspace::Union{Nothing,Integer}=nothing) where {T}
    isempty(dx) && return dx
    spatial_rank = ndims(dx) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    dil = spatial_vector(dilation, spatial_rank)
    ctype = something(compute_type, conv_compute_type(T))
    cmode = graph_conv_mode(mode)
    size(dx, ndims(dx)-1) == size(w, ndims(w)-1) * group ||
        throw(DimensionMismatch("input channels must equal filter channels times group"))
    key = convolution_dgrad_key(dx, dy, w, pre, post, str, dil, cmode, ctype,
                                 alpha, beta, deterministic, math_mode, max_workspace)
    try
        g = cached_graph(key) do
            build_convolution_dgrad_graph(dx, dy, w, pre, post, str, dil, cmode, ctype,
                                           alpha, beta; deterministic, math_mode,
                                           max_workspace)
        end
        execute!(g, tensor(g, "dY")=>dy, tensor(g, "W")=>w, tensor(g, "dX")=>dx)
        return dx
    catch e
        graph_unsupported(e) || rethrow()
    end
    if pre != post
        # no engine for asymmetric padding: compute into a padded dx and crop
        pad, padded_size, ranges = padded_convolution_geometry(dx, pre, post)
        dx_padded = similar(dx, padded_size)
        convolution_data_gradient!(dx_padded, dy, w; padding=pad, stride=str,
                                   dilation=dil, group, mode, alpha, compute_type=ctype,
                                   deterministic, math_mode, max_workspace)
        interior = view(dx_padded, ranges...)
        beta == 0 ? copyto!(dx, interior) : (dx .= interior .+ beta .* dx)
        return dx
    end
    pad = symmetric_padding(padding, spatial_rank)
    desc = convolution_descriptor(dx, pad, stride, dilation, mode, group, ctype)
    alpha, beta = scalingParameter(T, alpha), scalingParameter(T, beta)
    xdesc, ydesc, wdesc = cudnnTensorDescriptor(dx), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(w)
    perf = cudnnConvolutionBwdDataAlgoPerf(wdesc, w, ydesc, dy, desc, xdesc, dx, beta[] != 0)
    with_workspace(perf.memory) do workspace
        cudnnConvolutionBackwardData(handle(), alpha, wdesc, w, ydesc, dy, desc, perf.algo,
                                     workspace, sizeof(workspace), beta, xdesc, dx)
    end
    return dx
end

function convolution_filter_gradient!(dw::DenseCuArray{T}, x::DenseCuArray{T},
                                      dy::DenseCuArray{T};
                                      padding=0, stride=1, dilation=1,
                                      groups::Integer=1, group::Integer=groups,
                                      mode=:cross_correlation, alpha::Real=1,
                                      beta::Real=0, compute_type=nothing,
                                      deterministic::Bool=false,
                                      math_mode=CUDACore.math_mode(),
                                      max_workspace::Union{Nothing,Integer}=nothing) where {T}
    isempty(dw) && return dw
    spatial_rank = ndims(x) - 2
    pre, post = convolution_padding(padding, spatial_rank)
    str = spatial_vector(stride, spatial_rank)
    dil = spatial_vector(dilation, spatial_rank)
    ctype = something(compute_type, conv_compute_type(T))
    cmode = graph_conv_mode(mode)
    size(x, ndims(x)-1) == size(dw, ndims(dw)-1) * group ||
        throw(DimensionMismatch("input channels must equal filter channels times group"))
    if spatial_rank == 1 && pre != post
        # cuDNN 9.24 ignores 1D bwd-filter post-padding.
        padded_x, pad = padded_convolution_input(x, pre, post)
        return convolution_filter_gradient!(dw, padded_x, dy; padding=pad, stride=str,
                                            dilation=dil, groups=group, mode=cmode,
                                            alpha, beta, compute_type=ctype,
                                            deterministic, math_mode, max_workspace)
    end
    key = convolution_wgrad_key(dw, x, dy, pre, post, str, dil, cmode, ctype,
                                 alpha, beta, deterministic, math_mode, max_workspace)
    try
        g = cached_graph(key) do
            build_convolution_wgrad_graph(dw, x, dy, pre, post, str, dil, cmode, ctype,
                                           alpha, beta; deterministic, math_mode,
                                           max_workspace)
        end
        execute!(g, tensor(g, "X")=>x, tensor(g, "dY")=>dy, tensor(g, "dW")=>dw)
        return dw
    catch e
        graph_unsupported(e) || rethrow()
    end
    if pre != post
        # no engine for asymmetric padding: pad the input manually and retry
        padded_x, pad = padded_convolution_input(x, pre, post)
        return convolution_filter_gradient!(dw, padded_x, dy; padding=pad, stride=str,
                                            dilation=dil, group, mode, alpha, beta,
                                            compute_type=ctype, deterministic,
                                            math_mode, max_workspace)
    end
    pad = symmetric_padding(padding, spatial_rank)
    desc = convolution_descriptor(x, pad, stride, dilation, mode, group, ctype)
    alpha, beta = scalingParameter(T, alpha), scalingParameter(T, beta)
    xdesc, ydesc, wdesc = cudnnTensorDescriptor(x), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(dw)
    perf = cudnnConvolutionBwdFilterAlgoPerf(xdesc, x, ydesc, dy, desc, wdesc, dw, beta[] != 0)
    with_workspace(perf.memory) do workspace
        cudnnConvolutionBackwardFilter(handle(), alpha, xdesc, x, ydesc, dy, desc, perf.algo,
                                       workspace, sizeof(workspace), beta, wdesc, dw)
    end
    return dw
end

@doc raw"""
    convolution!(y, x, w; padding=0, stride=1, dilation=1, groups=1,
                 mode=:cross_correlation, alpha=1, beta=0,
                 bias=nothing, activation=nothing, z=nothing, compute_type=nothing)
    convolution_data_gradient!(dx, dy, w; kwargs...)
    convolution_filter_gradient!(dw, x, dy; kwargs...)

Convolve `x` with the filter `w`, or compute the gradients with respect to the
convolution input or filter. Arrays are in Julia memory order: spatial dimensions first,
then channels, then batch, with `w` shaped `(spatial..., C_in ÷ groups, C_out)`.

Computes `alpha * conv(x, w) + beta * y`, or, with `z` given, applies `activation`
(`:relu`, `:tanh`, `:sigmoid`, or `:elu`) to `alpha * conv(x, w) + beta * z + bias`
as a fused graph.

`padding` accepts a scalar, one value per spatial dimension, or per-side
`(pre1, post1, pre2, post2, ...)` pairs; asymmetric padding is supported natively.
`mode=:convolution` flips the kernel. `compute_type` defaults to `Float32` for
`Float16`/`BFloat16` data. Engine selection can be constrained with the `deterministic`,
`math_mode`, and `max_workspace` keywords.
"""
convolution!, convolution_data_gradient!, convolution_filter_gradient!

@public convolution!, convolution_data_gradient!, convolution_filter_gradient!
