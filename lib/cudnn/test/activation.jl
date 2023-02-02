using cuDNN:
    cudnnActivationForward,
    cudnnActivationForward!,
    cudnnActivationBackward,
    cudnnActivationDescriptor,
        cudnnActivationDescriptor_t,
        cudnnCreateActivationDescriptor,
        cudnnSetActivationDescriptor,
        cudnnGetActivationDescriptor,
        cudnnDestroyActivationDescriptor,
    cudnnActivationMode_t,
        CUDNN_ACTIVATION_SIGMOID,      # 0
        CUDNN_ACTIVATION_RELU,         # 1
        CUDNN_ACTIVATION_TANH,         # 2
        CUDNN_ACTIVATION_CLIPPED_RELU, # 3
        CUDNN_ACTIVATION_ELU,          # 4
        CUDNN_ACTIVATION_IDENTITY,     # 5
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN      # 1


@test cudnnActivationDescriptor(C_NULL) isa cudnnActivationDescriptor
@test Base.unsafe_convert(Ptr, cudnnActivationDescriptor(C_NULL)) isa Ptr
@test cudnnActivationDescriptor(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0) isa cudnnActivationDescriptor

(ax,ay) = randn.((10,10))
(cx,cy) = CuArray.((ax,ay))

function activationtest(;
    mode=CUDNN_ACTIVATION_SIGMOID,
    nanOpt=CUDNN_NOT_PROPAGATE_NAN,
    coef=1,
    alpha=1,
    beta=0,
)
    fx = (mode === CUDNN_ACTIVATION_SIGMOID ? 1 ./ (1 .+ exp.(-ax)) :
            mode === CUDNN_ACTIVATION_RELU ? max.(0,ax) :
            mode === CUDNN_ACTIVATION_TANH ? tanh.(ax) :
            mode === CUDNN_ACTIVATION_CLIPPED_RELU ? clamp.(ax,0,coef) :
            mode === CUDNN_ACTIVATION_ELU ? (x->(x >= 0 ? x : coef*(exp(x)-1))).(ax) :
            error("Unknown activation"))
    d = cudnnActivationDescriptor(mode,nanOpt,Cfloat(coef))
    y0 = alpha * fx
    y1 = y0 .+ beta * ay
    @test y0 ≈ cudnnActivationForward(cx; mode, nanOpt, coef, alpha) |> Array
    @test y0 ≈ cudnnActivationForward(cx, d; alpha) |> Array
    @test y1 ≈ cudnnActivationForward!(copy(cy), cx; mode, nanOpt, coef, alpha, beta) |> Array
    @test y1 ≈ cudnnActivationForward!(copy(cy), cx, d; alpha, beta) |> Array
end

activationtest(mode=CUDNN_ACTIVATION_SIGMOID)
activationtest(mode=CUDNN_ACTIVATION_RELU)
activationtest(mode=CUDNN_ACTIVATION_TANH)
activationtest(mode=CUDNN_ACTIVATION_CLIPPED_RELU)
activationtest(mode=CUDNN_ACTIVATION_ELU)
activationtest(nanOpt=CUDNN_PROPAGATE_NAN)
activationtest(coef=2,mode=CUDNN_ACTIVATION_CLIPPED_RELU)
activationtest(coef=2,mode=CUDNN_ACTIVATION_ELU)
activationtest(alpha=2)
activationtest(beta=2)
