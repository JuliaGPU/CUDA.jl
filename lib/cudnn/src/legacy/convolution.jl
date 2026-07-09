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
* `mathType = cuDNN.math_mode()`: whether or not the use of tensor op is permitted
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
    p = cudnnConvolutionFwdAlgoPerf(xDesc, x, wDesc, w, convDesc, yDesc, y, biasDesc, activation, beta!=0)
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
