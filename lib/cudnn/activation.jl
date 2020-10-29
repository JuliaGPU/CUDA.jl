# descriptor

mutable struct ActivationDesc
    ptr::cudnnActivationDescriptor_t
end

unsafe_free!(ad::ActivationDesc)=cudnnDestroyActivationDescriptor(ad.ptr)

Base.unsafe_convert(::Type{cudnnActivationDescriptor_t}, ad::ActivationDesc)=ad.ptr

function ActivationDesc(mode, coeff, reluNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    ad = Ref{cudnnActivationDescriptor_t}()
    cudnnCreateActivationDescriptor(ad)
    cudnnSetActivationDescriptor(ad[],mode,reluNanOpt,coeff)
    this = ActivationDesc(ad[])
    finalizer(unsafe_free!, this)
    return this
end


# wrappers

function cudnnActivationForward(x::DenseCuArray{T,N}, y::DenseCuArray{T,N}=x;
                                mode=CUDNN_ACTIVATION_RELU, # CUDNN_ACTIVATION_IDENTITY will not work
                                coeff=0.0, reluNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1, beta=0) where {T,N}
    cudnnActivationForward(handle(), ActivationDesc(mode, T(coeff), reluNanOpt),
                           scalingParameter(T, alpha), TensorDesc(x), x,
                           scalingParameter(T, beta ), TensorDesc(y), y)
    return  y
end

function cudnnActivationBackward(x::DenseCuArray{T,N}, dx::DenseCuArray{T,N}, y::DenseCuArray{T,N}, dy::DenseCuArray{T,N}=dx;
                                 mode=CUDNN_ACTIVATION_RELU, # CUDNN_ACTIVATION_IDENTITY will not work
                                 coeff=0.0, reluNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1, beta=0) where {T,N}
    cudnnActivationBackward(handle(), ActivationDesc(mode, T(coeff), reluNanOpt),
                            scalingParameter(T, alpha), TensorDesc( y),  y,
                            TensorDesc(dy), dy,
                            TensorDesc( x),  x,
                            scalingParameter(T, beta ), TensorDesc(dx), dx)
    return dx
end
