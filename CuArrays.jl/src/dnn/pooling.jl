using NNlib: PoolDims


# descriptor

mutable struct PoolDesc
    ptr::cudnnPoolingDescriptor_t
end

unsafe_free!(pd::PoolDesc)=cudnnDestroyPoolingDescriptor(pd.ptr)

Base.unsafe_convert(::Type{cudnnPoolingDescriptor_t}, pd::PoolDesc)=pd.ptr

function PoolDesc(nd, window, padding, stride, mode, maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    pd = Ref{cudnnPoolingDescriptor_t}()
    cudnnCreatePoolingDescriptor(pd)
    cudnnSetPoolingNdDescriptor(pd[],cudnnPoolingMode_t(mode),maxpoolingNanOpt,nd,pdsize(window,nd),pdsize(padding,nd),pdsize(stride,nd))
    this = PoolDesc(pd[])
    finalizer(unsafe_free!, this)
    return this
end

function PoolDesc(pdims::PoolDims, mode, maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    pd = NNlib.padding(pdims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn("CuDNN does not support asymmetric padding; defaulting to symmetric choice")
    end
    return PoolDesc(NNlib.spatial_dims(pdims), NNlib.kernel_size(pdims), pd[1:2:end],
                    NNlib.stride(pdims), mode, maxpoolingNanOpt)
end


# wrappers

function cudnnPoolingForward(y::CuArray{T,N}, x::CuArray{T,N}, pdims::PoolDims;
                             alpha=1, mode=0) where {T,N}
    beta = 0
    cudnnPoolingForward(handle(), PoolDesc(pdims, mode), Ref(T(alpha)), TensorDesc(x), x, Ref(T(beta)), TensorDesc(y), y)
    return y
end

function cudnnPoolingBackward(dx::CuArray{T,N}, dy::CuArray{T,N}, x::CuArray{T,N}, y::CuArray{T,N},
                              pdims::PoolDims; alpha=1, mode=0) where {T,N}
    if alpha!=1 && mode==0; error("Gradient of pool(alpha!=1,mode=0) broken in CUDNN"); end
    beta = 0
    cudnnPoolingBackward(handle(),
        PoolDesc(pdims, mode), Ref(T(alpha)), TensorDesc(y), y,
        TensorDesc(dy), dy, TensorDesc(x), x, Ref(T(beta)), TensorDesc(dx), dx
    )
    return dx
end
