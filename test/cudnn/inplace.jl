using Test, CUDA, Random
import CUDA.CUDNN:
    cudnnSetTensor!,
    cudnnScaleTensor!,
    cudnnScaleTensor,
    cudnnAddTensor!,
    cudnnAddTensor,
    CUDNN_TENSOR_NHWC


@testset "cudnn/inplace" begin
    x = CUDA.rand(10)
    cudnnSetTensor!(x, 7)
    @test all(isequal(7), Array(x))
    ax = rand(10)
    cx = CuArray(ax)
    @test 7*ax ≈ cudnnScaleTensor(cx, 7) |> Array
    @test 7*ax ≈ cudnnScaleTensor!(similar(cx), cx, 7) |> Array
    ax,ab = rand(5,4,3,2),rand(1,1,3,1)
    cx,cb = CuArray.((ax,ab))
    @test ax .+ ab ≈ cudnnAddTensor(cx, cb) |> Array
    @test ax .+ 7*ab ≈ cudnnAddTensor(cx, cb, alpha=7) |> Array
    @test 7*ax .+ ab ≈ cudnnAddTensor(cx, cb, beta=7) |> Array
    @test ax .+ ab ≈ cudnnAddTensor!(similar(cx), cx, cb) |> Array
    @test ax .+ 7*ab ≈ cudnnAddTensor!(similar(cx), cx, cb, alpha=7) |> Array
    @test 7*ax .+ ab ≈ cudnnAddTensor!(similar(cx), cx, cb, beta=7) |> Array
    @test ax .+ ab ≈ cudnnAddTensor!(cx, cx, cb) |> Array
    @test ax .+ ab ≈ cx |> Array
    ax,ab = rand(3,5,4,2),rand(3,1,1,1)
    cx,cb = CuArray.((ax,ab))
    @test ax .+ ab ≈ cudnnAddTensor(cx, cb, format=CUDNN_TENSOR_NHWC) |> Array
end
