using CUDA, Test, Random
using CUDA.CUDNN:
    cudnnOpTensor,
    cudnnOpTensor!,
    cudnnOpTensorDescriptor,
        cudnnOpTensorDescriptor_t,
        cudnnCreateOpTensorDescriptor,
        cudnnSetOpTensorDescriptor,
        cudnnGetOpTensorDescriptor,
        cudnnDestroyOpTensorDescriptor,
    cudnnOpTensorOp_t,
        CUDNN_OP_TENSOR_ADD,  # 0,
        CUDNN_OP_TENSOR_MUL,  # 1,
        CUDNN_OP_TENSOR_MIN,  # 2,
        CUDNN_OP_TENSOR_MAX,  # 3,
        CUDNN_OP_TENSOR_SQRT, # 4, performed only on first arg
        CUDNN_OP_TENSOR_NOT,  # 5, performed only on first arg
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    cudnnDataType,
    handle


@testset "cudnn/optensor" begin

    @test cudnnOpTensorDescriptor(C_NULL) isa cudnnOpTensorDescriptor
    @test Base.unsafe_convert(Ptr, cudnnOpTensorDescriptor(C_NULL)) isa Ptr
    @test cudnnOpTensorDescriptor(CUDNN_OP_TENSOR_ADD,cudnnDataType(Float32),CUDNN_NOT_PROPAGATE_NAN) isa cudnnOpTensorDescriptor

    (ax1,ax2,ay) = rand.((10,10,10))
    (cx1,cx2,cy) = CuArray.((ax1,ax2,ay))

    function optensortest(
        ;op=CUDNN_OP_TENSOR_ADD,
        nanOpt=CUDNN_NOT_PROPAGATE_NAN,
        compType=(eltype(ax1) <: Float64 ? Float64 : Float32),
        alpha1=1,
        alpha2=1,
        beta=0,
    )
        f1 = (op === CUDNN_OP_TENSOR_ADD ? alpha1*ax1 .+ alpha2*ax2 :
              op === CUDNN_OP_TENSOR_MUL ? (alpha1*ax1) .* (alpha2*ax2) :
              op === CUDNN_OP_TENSOR_MIN ? min.(alpha1*ax1, alpha2*ax2) :
              op === CUDNN_OP_TENSOR_MAX ? max.(alpha1*ax1, alpha2*ax2) :
              op === CUDNN_OP_TENSOR_SQRT ? sqrt.(alpha1*ax1) :
              op === CUDNN_OP_TENSOR_NOT ? 1 .- ax1 :
              error("Unknown optensor"))
        f2 = f1 .+ beta * ay
        d = cudnnOpTensorDescriptor(op,cudnnDataType(compType),nanOpt)
        ((f1 ≈ cudnnOpTensor(cx1, cx2; op, compType, nanOpt, alpha1, alpha2) |> Array) &&
         (f1 ≈ cudnnOpTensor(cx1, cx2, d; alpha1, alpha2) |> Array) &&
         (f2 ≈ cudnnOpTensor!(copy(cy), cx1, cx2; op, compType, nanOpt, alpha1, alpha2, beta) |> Array) &&
         (f2 ≈ cudnnOpTensor!(copy(cy), cx1, cx2, d; alpha1, alpha2, beta) |> Array))
    end
    
    @test optensortest(op = CUDNN_OP_TENSOR_ADD)
    @test optensortest(op = CUDNN_OP_TENSOR_MUL)
    @test optensortest(op = CUDNN_OP_TENSOR_MIN)
    @test optensortest(op = CUDNN_OP_TENSOR_MAX)
    @test optensortest(op = CUDNN_OP_TENSOR_SQRT)
    @test optensortest(op = CUDNN_OP_TENSOR_NOT)
    @test optensortest(nanOpt = CUDNN_PROPAGATE_NAN)
    @test optensortest(alpha1 = 2)
    @test optensortest(alpha2 = 2)
    @test optensortest(beta = 2)
end
