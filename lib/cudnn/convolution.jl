# There is too much common code between cudnnConvolutionForward and cudnnConvolutionBiasActivationForward.
# We will have a single interface and call one or the other based on keyword arguments.

"""
    cudnnConvolutionForward(w, x; bias, activation, mode, padding, stride, dilation, group, mathType, reorderType, alpha, beta, z, format)
    cudnnConvolutionForward(w, x, d::cudnnConvolutionDescriptor; bias, activation, alpha, beta, z, format)
    cudnnConvolutionForward!(y, w, x; bias, activation, mode, padding, stride, dilation, group, mathType, reorderType, alpha, beta, z, format)
    cudnnConvolutionForward!(y, w, x, d::cudnnConvolutionDescriptor; bias, activation, alpha, beta, z, format)

Return the convolution of filter `w` with tensor `x`, overwriting `y` if provided, according
to keyword arguments or the convolution descriptor `d`. Optionally perform bias addition,
activation and/or scaling:

    y .= activation.(alpha * conv(w,x) + beta * z .+ bias)

All tensors should have the same number of dimensions. If they are less than 4-D their
dimensions are assumed to be padded on the left with ones. `x` has size `(X...,Cx,N)` where
`(X...)` are the spatial dimensions, `Cx` is the number of input channels, and `N` is the
number of instances. `y,z` have size `(Y...,Cy,N)` where `(Y...)` are the spatial dimensions
and `Cy` is the number of output channels (`y` and `z` can be the same array). Both `Cx` and
`Cy` have to be an exact multiple of `group`.  `w` has size `(W...,Cx÷group,Cy)` where
`(W...)` are the filter dimensions. `bias` has size `(1...,Cy,1)`.

The arguments `padding`, `stride` and `dilation` can be specified as `n-2` dimensional
vectors, tuples or a single integer which is assumed to be repeated `n-2` times. If any of
the entries is larger than the corresponding `x` dimension, the `x` dimension is used
instead. For a description of different types of convolution see:
https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

Keyword arguments:
* `activation = CUDNN_ACTIVATION_IDENTITY`: the only other supported option is `CUDNN_ACTIVATION_RELU`
* `bias = nothing`: add bias if provided
* `z = nothing`: add `beta*z`, `z` can be `nothing`, `y` or another array similar to `y`
* `alpha = 1, beta = 0`: scaling parameters
* `format = CUDNN_TENSOR_NCHW`: order of tensor dimensions, the other alternative is `CUDNN_TENSOR_NHWC`. Note that Julia dimensions will have the opposite order, i.e. WHCN or CWHN.

Keyword arguments describing the convolution when `d` is not given:
* `mode = CUDNN_CONVOLUTION`: alternatively `CUDNN_CROSS_CORRELATION`
* `padding = 0`: padding assumed around `x`
* `stride = 1`: how far to shift the convolution window at each step
* `dilation = 1`: dilation factor
* `group = 1`: number of groups to be used
* `mathType = CUDNN.math_mode()`: whether or not the use of tensor op is permitted
* `reorderType = CUDNN_DEFAULT_REORDER`: convolution reorder type
"""
cudnnConvolutionForward, cudnnConvolutionForward!


# Public methods
cudnnConvolutionForward(w, x; o...)     = cudnnConvolutionForwardWithDefaults(w, x; o...)
cudnnConvolutionForward!(y, w, x; o...) = cudnnConvolutionForwardWithDefaults(w, x; y, o...)
cudnnConvolutionForward(w, x, d::cudnnConvolutionDescriptor; o...)     = cudnnConvolutionForwardWithDefaults(w, x; convDesc=d, o...)
cudnnConvolutionForward!(y, w, x, d::cudnnConvolutionDescriptor; o...) = cudnnConvolutionForwardWithDefaults(w, x; y, convDesc=d, o...)


# Private method
function cudnnConvolutionForwardWithDefaults(
    w, x;

    # convDesc arguments
    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 0,  # >= 0
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1,   # >= 1
    dilation::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1, # >= 1
    mode::cudnnConvolutionMode_t = CUDNN_CONVOLUTION,
    mathType::cudnnMathType_t = math_mode(),
    reorderType::cudnnReorderType_t = CUDNN_DEFAULT_REORDER,  # related to cudnnReorderFilterAndBias?
    group::Integer = 1,
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    convDesc::cudnnConvolutionDescriptor = cudnnConvolutionDescriptor(convdims(padding,size(x),0), convdims(stride,size(x),1), convdims(dilation,size(x),1), mode, cudnnDataType(eltype(x)), mathType, reorderType, Cint(group)),

    # output array, descriptors, scaling factors
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format),
    wDesc::cudnnFilterDescriptor = cudnnFilterDescriptor(w; format),
    y = cudnnConvolutionForwardOutput(x, xDesc, wDesc, convDesc, format),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y; format),
    alpha::Real = 1,
    beta::Real = 0,

    # convbiasact arguments
    bias = nothing,
    z = nothing,
    biasDesc::Union{Nothing,cudnnTensorDescriptor} = (bias===nothing ? nothing : cudnnTensorDescriptor(bias; format)),
    zDesc::Union{Nothing,cudnnTensorDescriptor} = (z === nothing ? nothing : cudnnTensorDescriptor(z; format)),
    activation::cudnnActivationMode_t = CUDNN_ACTIVATION_IDENTITY, # coef and nanOpt are not useful options for convbiasact which only supports relu

    # gradient buffers
    dw = Ref{Any}(nothing),
    dx = Ref{Any}(nothing),
    dz = Ref{Any}(nothing),
    dbias = Ref{Any}(nothing),
)
    T = eltype(x)
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    # Backward called separately on each variable. We will calculate all gradients on first call. Use `dready` to avoid subsequent calls.
    dready = Ref{Bool}(false)   # this will be turned to `true` by the first backward call.
    cudnnConvolutionForwardAD(w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready)
end


# AD method
function cudnnConvolutionForwardAD(w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready)
    p = cudnnConvolutionFwdAlgoPerf(xDesc, x, wDesc, w, convDesc, yDesc, y, biasDesc, activation)
    with_workspace(p.memory) do workspace
        if bias === nothing && activation === CUDNN_ACTIVATION_IDENTITY && (z === y || beta[] == 0)
            cudnnConvolutionForward(handle(), alpha, xDesc, x, wDesc, w, convDesc, p.algo, workspace, sizeof(workspace), beta, yDesc, y)
        else
            @assert activation === CUDNN_ACTIVATION_IDENTITY || activation === CUDNN_ACTIVATION_RELU  "Only RELU and IDENTITY supported"
            activationDesc = cudnnActivationDescriptor(activation, CUDNN_NOT_PROPAGATE_NAN, Cdouble(1.0))
            # bias and z cannot be null for cudnnConvolutionBiasActivationForward
            if z === nothing; z, zDesc = y, yDesc; beta[] = 0; end
            if bias === nothing
                format = cudnnGetFilterDescriptor(wDesc)[3]
                bdim = (format === CUDNN_TENSOR_NHWC ? 1 : ndims(y)-1)
                bias = fill!(similar(w, ntuple(i->(i==bdim ? size(y,i) : 1), ndims(y))), 0)
                biasDesc = cudnnTensorDescriptor(bias; format)
            end
            cudnnConvolutionBiasActivationForward(handle(), alpha, xDesc, x, wDesc, w, convDesc, p.algo, workspace, sizeof(workspace), beta, zDesc, z, biasDesc, bias, activationDesc, yDesc, y)
        end
    end
    return y
end

# Helper for cudnnConvolutionDescriptor
function cudnnSetConvolutionDescriptor(
    ptr::cudnnConvolutionDescriptor_t,
    padding::Vector{Cint},
    stride::Vector{Cint},
    dilation::Vector{Cint},
    mode::cudnnConvolutionMode_t,
    dataType::cudnnDataType_t,
    mathType::cudnnMathType_t,
    reorderType::cudnnReorderType_t,
    groupCount::Cint,
)
    cudnnSetConvolutionNdDescriptor(ptr, Cint(length(padding)), padding, stride, dilation, mode, dataType)
    mathType != CUDNN_DEFAULT_MATH       && cudnnSetConvolutionMathType(ptr, mathType)
    reorderType != CUDNN_DEFAULT_REORDER && cudnnSetConvolutionReorderType(ptr, reorderType)
    groupCount != 1                      && cudnnSetConvolutionGroupCount(ptr, groupCount)
end


## cudnnConvolutionForward helpers:

function cudnnConvolutionForwardOutput(x, xDesc, wDesc, convDesc, format)
    d = Array{Cint}(undef, max(4, ndims(x))) # d = [N,C,Yn,...,Y1] no matter what format
    cudnnGetConvolutionNdForwardOutputDim(convDesc, xDesc, wDesc, length(d), d)
    if length(d) > ndims(x) # This happens when x is (X,C,N), xDesc is [N,C,X,1]
        @assert all(d[ndims(x)+1:end] .== 1)
        d = d[1:ndims(x)]
    end
    # ydims(NCHW)=(Y1,...,Yn,C,N) ydims(NHWC)=(C,Y1,...,Yn,N)
    ydims = (format === CUDNN_TENSOR_NCHW ? reverse(d) : (d[2],d[end:-1:3]...,d[1]))
    similar(x, ydims...)
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


## Utilities to find a fast algorithm

const cudnnConvolutionFwdAlgoPerfCache = Dict{Tuple,cudnnConvolutionFwdAlgoPerf_t}()
const cudnnConvolutionFwdAlgoPerfCacheLock = ReentrantLock()
function cudnnConvolutionFwdAlgoPerf(xDesc, x, wDesc, w, convDesc, yDesc, y, biasDesc, activation)
    key = (xDesc, wDesc, convDesc, biasDesc, activation)
    val = lock(cudnnConvolutionFwdAlgoPerfCacheLock) do
         get(cudnnConvolutionFwdAlgoPerfCache, key, nothing)
    end
    if val === nothing
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionFwdAlgoPerf_t}(undef,requestedAlgoCount)
        workspaceSize() = cudnnFindConvolutionAlgorithmWorkspaceSize(x)
        with_workspace(workspaceSize) do workspace
            cudnnFindConvolutionForwardAlgorithmEx(handle(),xDesc,x,wDesc,w,convDesc,yDesc,y,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,sizeof(workspace))
        end
        val = cudnnConvolutionAlgoPerfChoose(perfResults, returnedAlgoCount[1])
        lock(cudnnConvolutionFwdAlgoPerfCacheLock) do
            cudnnConvolutionFwdAlgoPerfCache[key] = val
        end
    end
    return val
end

const cudnnConvolutionBwdDataAlgoPerfCache = Dict{Tuple,cudnnConvolutionBwdDataAlgoPerf_t}()
const cudnnConvolutionBwdDataAlgoPerfCacheLock = ReentrantLock()
function cudnnConvolutionBwdDataAlgoPerf(wDesc, w, dyDesc, dy, convDesc, dxDesc, dx)
    key = (wDesc, dyDesc, convDesc)
    val = lock(cudnnConvolutionBwdDataAlgoPerfCacheLock) do
        get(cudnnConvolutionBwdDataAlgoPerfCache, key, nothing)
    end
    if val === nothing
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionBwdDataAlgoPerf_t}(undef,requestedAlgoCount)
        workspaceSize() = cudnnFindConvolutionAlgorithmWorkspaceSize(dx)
        with_workspace(workspaceSize) do workspace
            cudnnFindConvolutionBackwardDataAlgorithmEx(handle(),wDesc,w,dyDesc,dy,convDesc,dxDesc,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,sizeof(workspace))
        end
        val = cudnnConvolutionAlgoPerfChoose(perfResults, returnedAlgoCount[1])
        lock(cudnnConvolutionBwdDataAlgoPerfCacheLock) do
            cudnnConvolutionBwdDataAlgoPerfCache[key] = val
        end
    end
    val
end

const cudnnConvolutionBwdFilterAlgoPerfCache = Dict{Tuple,cudnnConvolutionBwdFilterAlgoPerf_t}()
const cudnnConvolutionBwdFilterAlgoPerfCacheLock = ReentrantLock()
function cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, dyDesc, dy, convDesc, dwDesc, dw)
    key = (xDesc, dyDesc, convDesc)
    val = lock(cudnnConvolutionBwdFilterAlgoPerfCacheLock) do
        get(cudnnConvolutionBwdFilterAlgoPerfCache, (xDesc, dyDesc, convDesc), nothing)
    end
    if val === nothing
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionBwdFilterAlgoPerf_t}(undef,requestedAlgoCount)
        workspaceSize() = cudnnFindConvolutionAlgorithmWorkspaceSize(x)
        with_workspace(workspaceSize) do workspace
            cudnnFindConvolutionBackwardFilterAlgorithmEx(handle(),xDesc,x,dyDesc,dy,convDesc,dwDesc,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,sizeof(workspace))
        end
        val = cudnnConvolutionAlgoPerfChoose(perfResults, returnedAlgoCount[1])
        lock(cudnnConvolutionBwdFilterAlgoPerfCacheLock) do
            cudnnConvolutionBwdFilterAlgoPerfCache[key] = val
        end
    end
    val
end


# Return algorithm with best memory that is within 10% of best time
function cudnnConvolutionAlgoPerfChoose(ps, n)
    (ibest,mbest,tbest) = (0,Inf,Inf)
    for i in 1:n
        # These metrics are written in a sorted fashion where the first element has the lowest compute time.
        if ps[i].status == CUDNN_STATUS_SUCCESS && ps[i].memory < mbest && ps[i].time < tbest * 1.1
            (ibest,mbest,tbest) = (i,ps[i].memory,ps[i].time)
        end
    end
    if ibest == 0
        @warn "No valid algorithm found, probably bad params for convolution." maxlog=1
        ibest = findfirst(p->p.algo==0, ps)
        ibest === nothing && error("Cannot find backup algorithm for convolution, giving up.")
    end
    return ps[ibest]
end


# Allocate the maximum reasonable amount of memory for algorithm discovery
function cudnnFindConvolutionAlgorithmWorkspaceSize(x)
    gpufree = CUDA.available_memory() + coalesce(CUDA.cached_memory(), 0)
    min(gpufree ÷ 10, sizeof(x) * 100)
end
