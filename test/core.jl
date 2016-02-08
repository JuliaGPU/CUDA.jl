@testset "CUDA.jl core" begin

dev = CuDevice(0)
ctx = CuContext(dev)


@testset "API call wrapper" begin

    @cucall(:cuDriverGetVersion, (Ptr{Cint},), Ref{Cint}())

    @test_throws ErrorException @cucall(:nonExisting, ())
    CUDA.trace(prefix=" ")

    @test_throws ErrorException eval(
        quote
            foo = :bar
            @cucall(foo, ())
        end
    )

    @test_throws CuError @cucall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), Ref{Ptr{Void}}(), 0)

end


@testset "compilation & execution" begin

    @compile dev reference_dummy """
    __global__ void reference_dummy()
    {
    }
    """

    CUDA.launch(reference_dummy(), 1, 1, ())

end


@testset "argument passing" begin

    dims = (16, 16)
    len = prod(dims)

    @compile dev reference_copy """
    __global__ void reference_copy(const float *input, float *output)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        output[i] = input[i];
    }
    """

    let
        input = round(rand(Float32, dims) * 100)

        input_dev = CuArray(input)
        output_dev = CuArray(Float32, dims)

        CUDA.launch(reference_copy(), len, 1, (input_dev.ptr, output_dev.ptr))
        output = to_host(output_dev)
        @test_approx_eq input output

        free(input_dev)
        free(output_dev)
    end

end


destroy(ctx)

end