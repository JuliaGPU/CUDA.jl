using CUDA, Test, Random, Statistics
using CUDA.CUDNN:
    cudnnReduceTensor,
    cudnnReduceTensor!,
    cudnnGetReductionIndicesSize,
    cudnnGetReductionWorkspaceSize,
    cudnnReduceTensorDescriptor,
        cudnnReduceTensorDescriptor_t,
        cudnnCreateReduceTensorDescriptor,
        cudnnSetReduceTensorDescriptor,
        cudnnGetReduceTensorDescriptor,
        cudnnDestroyReduceTensorDescriptor,
    cudnnReduceTensorOp_t,
        CUDNN_REDUCE_TENSOR_ADD,          # 0,
        CUDNN_REDUCE_TENSOR_MUL,          # 1,
        CUDNN_REDUCE_TENSOR_MIN,          # 2,
        CUDNN_REDUCE_TENSOR_MAX,          # 3,
        CUDNN_REDUCE_TENSOR_AMAX,         # 4,
        CUDNN_REDUCE_TENSOR_AVG,          # 5,
        CUDNN_REDUCE_TENSOR_NORM1,        # 6,
        CUDNN_REDUCE_TENSOR_NORM2,        # 7,
        CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS, # 8,
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    cudnnReduceTensorIndices,
    cudnnReduceTensorIndices_t,
        CUDNN_REDUCE_TENSOR_NO_INDICES,        # 0,
        CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, # 1,
    cudnnIndicesType,
    cudnnIndicesType_t,
        CUDNN_32BIT_INDICES, # 0,
        CUDNN_64BIT_INDICES, # 1,
        CUDNN_16BIT_INDICES, # 2,
        CUDNN_8BIT_INDICES,  # 3,
    cudnnDataType,
    handle


@testset "cudnn/reduce" begin

    @test cudnnReduceTensorDescriptor(C_NULL) isa cudnnReduceTensorDescriptor
    @test Base.unsafe_convert(Ptr, cudnnReduceTensorDescriptor(C_NULL)) isa Ptr
    @test cudnnReduceTensorDescriptor(CUDNN_REDUCE_TENSOR_ADD,cudnnDataType(Float32),CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES) isa cudnnReduceTensorDescriptor

    (ax,ay) = randn(Float32,10,10), randn(Float32,10,1)
    (cx,cy) = CuArray.((ax,ay))

    function reducetensortest(
        ; op::cudnnReduceTensorOp_t = CUDNN_REDUCE_TENSOR_ADD,
        compType::DataType = (eltype(ax) <: Float64 ? Float64 : Float32),
        nanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
        indices::Union{Vector{<:Unsigned},Nothing} = nothing,
        d::cudnnReduceTensorDescriptor = cudnnReduceTensorDescriptor(op, cudnnDataType(compType), nanOpt, cudnnReduceTensorIndices(op, indices), cudnnIndicesType(indices)),
        alpha::Real = 1,
        beta::Real = 0,
    )
        f0 = (op === CUDNN_REDUCE_TENSOR_ADD          ? sum(ax, dims=2) :
              op === CUDNN_REDUCE_TENSOR_MUL          ? prod(ax, dims=2) :
              op === CUDNN_REDUCE_TENSOR_MIN          ? minimum(ax, dims=2) :
              op === CUDNN_REDUCE_TENSOR_MAX          ? maximum(ax, dims=2) :
              op === CUDNN_REDUCE_TENSOR_AMAX         ? maximum(abs, ax, dims=2) :
              op === CUDNN_REDUCE_TENSOR_AVG          ? mean(ax, dims=2) :
              op === CUDNN_REDUCE_TENSOR_NORM1        ? sum(abs, ax, dims=2) :
              op === CUDNN_REDUCE_TENSOR_NORM2        ? sqrt.(sum(abs2, ax, dims=2)) :
              op === CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS ? (ax1=copy(ax);ax1[ax.==0].=1;prod(ax1,dims=2)) :
              error("Unknown reducetensor"))
        f1 = alpha * f0
        f2 = f1 + beta * ay
        dims = size(ay)
        ((f1 ≈ cudnnReduceTensor(cx; dims, op, compType, nanOpt, indices, alpha) |> Array) &&
         (f1 ≈ cudnnReduceTensor(cx, d; dims, indices, alpha) |> Array) &&
         (f2 ≈ cudnnReduceTensor!(copy(cy), cx; op, compType, nanOpt, indices, alpha, beta) |> Array) &&
         (f2 ≈ cudnnReduceTensor!(copy(cy), cx, d; indices, alpha, beta) |> Array))
    end
    
    @test reducetensortest()
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_MUL)
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_MIN)
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_MAX)
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_AMAX)
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_AVG)
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_NORM1)
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_NORM2)
    @test reducetensortest(op = CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
    @test reducetensortest(nanOpt = CUDNN_PROPAGATE_NAN)
    @test reducetensortest(alpha = 2)
    @test reducetensortest(beta = 2)
end
