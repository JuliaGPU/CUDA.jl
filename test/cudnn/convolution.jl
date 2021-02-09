using Test, CUDA, Random
import NNlib
using CUDA.CUDNN:
    cudnnConvolutionForward,
    cudnnConvolutionForward!,
    cudnnConvolutionBackwardFilter,
    cudnnConvolutionBackwardData,
    cudnnGetConvolutionNdForwardOutputDim,
    cudnnSetConvolutionMathType,
    cudnnSetConvolutionReorderType,
    cudnnSetConvolutionGroupCount,
    cudnnFindConvolutionForwardAlgorithmEx,
        cudnnConvolutionFwdAlgoPerf_t,
    cudnnFindConvolutionBackwardFilterAlgorithmEx,
        cudnnConvolutionBwdFilterAlgoPerf_t,
    cudnnFindConvolutionBackwardDataAlgorithmEx,
        cudnnConvolutionBwdDataAlgoPerf_t,
    cudnnConvolutionDescriptor,
        cudnnConvolutionDescriptor_t,
        cudnnCreateConvolutionDescriptor,
        cudnnSetConvolutionNdDescriptor,
        cudnnDestroyConvolutionDescriptor,
    cudnnConvolutionMode_t,
        CUDNN_CONVOLUTION,       # 0
        CUDNN_CROSS_CORRELATION, # 1
    cudnnActivationMode_t,
        CUDNN_ACTIVATION_SIGMOID,      # 0
        CUDNN_ACTIVATION_RELU,         # 1
        CUDNN_ACTIVATION_TANH,         # 2
        CUDNN_ACTIVATION_CLIPPED_RELU, # 3
        CUDNN_ACTIVATION_ELU,          # 4
        CUDNN_ACTIVATION_IDENTITY,     # 5
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    cudnnMathType_t,
        CUDNN_DEFAULT_MATH,                    # 0
        CUDNN_TENSOR_OP_MATH,                  # 1
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2
        CUDNN_FMA_MATH,                        # 3
    cudnnReorderType_t,
        CUDNN_DEFAULT_REORDER, # 0
        CUDNN_NO_REORDER,      # 1
    cudnnConvolutionFwdAlgo_t,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,         # 0
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, # 1
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,                  # 2
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,                # 3
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,                   # 4
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,            # 5
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,              # 6
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,     # 7
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT,                 # 8
    cudnnConvolutionBwdFilterAlgo_t,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,                 # 0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,                 # 1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,               # 2,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,                 # 3, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,          # 4, /* not implemented */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED, # 5,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,        # 6,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,             # 7
    cudnnConvolutionBwdDataAlgo_t,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,                 # 0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,                 # 1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,               # 2,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,        # 3,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,          # 4,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, # 5,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,             # 6
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    cudnnDataType,
    convdims,
    math_mode,
    handle

@testset "cudnn/convolution" begin
    T = Float32
    ax,aw,ab = randn(T,8,8,4,4),randn(T,3,3,4,4),randn(T,1,1,4,1)
    cx,cw,cb = CuArray.((ax,aw,ab))

    function convtest(;
                      blendz=false,
                      bias=nothing,
                      activation = CUDNN_ACTIVATION_IDENTITY,
                      mode = CUDNN_CONVOLUTION,
                      padding = 0,
                      stride = 1,
                      dilation = 1,
                      group = 1,
                      dataType = eltype(cx),
                      mathType = math_mode(),
                      reorderType = CUDNN_DEFAULT_REORDER,
                      alpha = 1,
                      beta = 0)
        if group == 1
            cdims = NNlib.DenseConvDims(ax, aw; stride, padding, dilation, flipkernel = (mode === CUDNN_CROSS_CORRELATION))
            ay = NNlib.conv(ax, aw, cdims)
            cw0 = cw
        else
            # Implement grouped convolution
            xchan = size(aw,3)÷group
            ychan = size(aw,4)÷group
            xdims = (size(ax,1),size(ax,2),xchan,size(ax,4))
            wdims = (size(aw,1),size(aw,2),xchan,ychan)
            cdims = NNlib.DenseConvDims(xdims, wdims; stride, padding, dilation, flipkernel = (mode === CUDNN_CROSS_CORRELATION))
            ay = nothing
            for g in 1:group
                xrange = 1+(g-1)*xchan:g*xchan
                yrange = 1+(g-1)*ychan:g*ychan
                ay0 = NNlib.conv(ax[:,:,xrange,:], aw[:,:,1:xchan,yrange], cdims)
                ay = (ay === nothing ? ay0 : cat(ay, ay0; dims=3))
            end
            cw0 = CuArray(aw[:,:,1:xchan,:])
        end

        if alpha != 1; ay = alpha * ay; end
        if bias != nothing; ay = ay .+ Array(bias); end

        act = (activation === CUDNN_ACTIVATION_RELU ? NNlib.relu :
               activation === CUDNN_ACTIVATION_IDENTITY ? identity :
               error("Unsupported activation $activation"))
        ay1 = act.(ay)

        az0 = randn(T,size(ay)...)
        ay0 = randn(T,size(ay)...)
        cy0, cy1 = CuArray.((ay0,ay0))
        if blendz
            cz0 = cz1 = CuArray(az0)
            ay2 = act.(ay .+ beta * az0)
        else
            cz0, cz1 = cy0, cy1
            ay2 = act.(ay .+ beta * ay0)
        end
        d = cudnnConvolutionDescriptor(convdims(padding,size(ax)), convdims(stride,size(ax)), convdims(dilation,size(ax)), mode, cudnnDataType(dataType), mathType, reorderType, Cint(group))
        ((ay1 ≈ cudnnConvolutionForward(cw0, cx; bias, activation, mode, padding, stride, dilation, group, mathType, reorderType, alpha) |> Array) &&
         (ay1 ≈ cudnnConvolutionForward(cw0, cx, d; bias, activation, alpha) |> Array) &&
         (ay2 ≈ cudnnConvolutionForward!(cy0, cw0, cx; z=cz0, bias, activation, mode, padding, stride, dilation, group, mathType, reorderType, alpha, beta) |> Array) &&
         (ay2 ≈ cudnnConvolutionForward!(cy1, cw0, cx, d; z=cz1, bias, activation, alpha, beta) |> Array))
    end

    # These call cudnnConvolutionForward
    @test convtest()
    @test convtest(padding=1)
    @test convtest(stride=2)
    @test convtest(dilation=2)
    @test convtest(group=2) # See https://blog.yani.ai/filter-group-tutorial/
    @test convtest(mathType=CUDNN_DEFAULT_MATH)
    @test convtest(mathType=CUDNN_TENSOR_OP_MATH)
    @test convtest(mathType=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test convtest(reorderType=CUDNN_NO_REORDER)
    @test convtest(alpha=2)
    @test convtest(beta=2)

    # These call cudnnConvolutionBiasActivationForward
    @test convtest(bias=cb)
    @test convtest(blendz=true)
    @test convtest(activation=CUDNN_ACTIVATION_RELU)
    @test convtest(bias=cb,blendz=true)
    @test convtest(bias=cb,activation=CUDNN_ACTIVATION_RELU)
    @test convtest(bias=cb,padding=1)
    @test convtest(bias=cb,stride=2)
    @test convtest(bias=cb,dilation=2)
    @test convtest(bias=cb,group=2)
    @test convtest(bias=cb,mathType=CUDNN_DEFAULT_MATH)
    @test convtest(bias=cb,mathType=CUDNN_TENSOR_OP_MATH)
    @test convtest(bias=cb,mathType=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test convtest(bias=cb,reorderType=CUDNN_NO_REORDER)
    @test convtest(bias=cb,alpha=2)
    @test convtest(bias=cb,beta=2)
    @test convtest(bias=cb,beta=2,blendz=true)

    # Test tensor format
    cx2,cw2,cb2 = (x->permutedims(x,(3,1,2,4))).((cx,cw,cb))
    whcn = cudnnConvolutionForward(cw,cx)
    cwhn = cudnnConvolutionForward(cw2,cx2,format=CUDNN_TENSOR_NHWC)
    @test cwhn ≈ permutedims(whcn,(3,1,2,4))
    whcn = cudnnConvolutionForward(cw,cx;bias=cb)
    cwhn = cudnnConvolutionForward(cw2,cx2;bias=cb2,format=CUDNN_TENSOR_NHWC)
    @test cwhn ≈ permutedims(whcn,(3,1,2,4))
end
