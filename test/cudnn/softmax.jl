using Test, CUDA
using CUDA.CUDNN:
    cudnnSoftmaxForward,
    cudnnSoftmaxForward!,
    cudnnSoftmaxBackward,
    cudnnSoftmaxAlgorithm_t,
        CUDNN_SOFTMAX_FAST,     # 0, /* straightforward implementation */
        CUDNN_SOFTMAX_ACCURATE, # 1, /* subtract max from every point to avoid overflow */
        CUDNN_SOFTMAX_LOG,      # 2
    cudnnSoftmaxMode_t,
        CUDNN_SOFTMAX_MODE_INSTANCE, # 0, /* compute the softmax over all C, H, W for each N */
        CUDNN_SOFTMAX_MODE_CHANNEL,  # 1  /* compute the softmax over all C for each H, W, N */
    handle


@testset "cudnn/softmax" begin
    ax,ay = randn(Float32,10,10),randn(Float32,10,10)
    cx,cy = CuArray.((ax,ay))

    function softmaxtest(
        ; alpha=1,
        beta=0,
        mode=CUDNN_SOFTMAX_MODE_INSTANCE,
        algo=CUDNN_SOFTMAX_FAST
    )
        d = mode === CUDNN_SOFTMAX_MODE_INSTANCE ? 1 : 2
        x = ax .- maximum(ax, dims=d)
        y = x .- log.(sum(exp.(x), dims=d))
        if algo !== CUDNN_SOFTMAX_LOG; y = exp.(y); end
        add1(x)=reshape(x, (size(x)..., 1))
        if mode === CUDNN_SOFTMAX_MODE_CHANNEL
            y,cx1,cy1 = add1.((y,cx,cy))
        else
            cx1,cy1 = cx,cy
        end
        y0 = alpha * y
        y1 = y0 .+ beta * ay
        ((y0 ≈ cudnnSoftmaxForward(cx1; algo, mode, alpha) |> Array) &&
         (y1 ≈ cudnnSoftmaxForward!(copy(cy1), cx1; algo, mode, alpha, beta) |> Array))
    end

    @test softmaxtest()
    @test softmaxtest(alpha=2)
    @test softmaxtest(beta=2)
    @test softmaxtest(mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test softmaxtest(mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test softmaxtest(algo=CUDNN_SOFTMAX_FAST)
    @test softmaxtest(algo=CUDNN_SOFTMAX_ACCURATE)
    @test softmaxtest(algo=CUDNN_SOFTMAX_LOG)
end
