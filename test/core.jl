@testset "CUDA.jl core" begin

dev = CuDevice(0)
ctx = CuContext(dev)


@testset "API call wrapper" begin
    @cucall(:cuDriverGetVersion, (Ptr{Cint},), Ref{Cint}())

    @test_throws ErrorException @cucall(:nonExisting, ())
    CUDAnative.trace(prefix=" ")

    @test_throws ErrorException eval(
        quote
            foo = :bar
            @cucall(foo, ())
        end
    )

    @test_throws CuError @cucall(:cuMemAlloc, (Ptr{Ptr{Void}}, Csize_t), Ref{Ptr{Void}}(), 0)
end


@testset "PTX loading & execution" begin
    md = CuModuleFile(joinpath(Base.source_dir(), "vectorops.ptx"))
    vadd = CuFunction(md, "vadd")
    vmul = CuFunction(md, "vmul")
    vsub = CuFunction(md, "vsub")
    vdiv = CuFunction(md, "vdiv")

    a = rand(Float32, 10)
    b = rand(Float32, 10)
    ad = CuArray(a)
    bd = CuArray(b)

    # Addition
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vadd, 10, 1, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test_approx_eq c a+b
        free(cd)
    end

    # Subtraction
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vsub, 10, 1, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test_approx_eq c a-b
        free(cd)
    end

    # Multiplication
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vmul, 10, 1, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test_approx_eq c a.*b
        free(cd)
    end

    # Division
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vdiv, 10, 1, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test_approx_eq c a./b
        free(cd)
    end

    free(ad)
    free(bd)
    unload(md)
end


@testset "CuArray" begin
    # Negative test cases
    a = rand(Float32, 10)
    ad = CuArray(Float32, 5)
    @test_throws ArgumentError copy!(ad, a)
    @test_throws ArgumentError copy!(a, ad)

    # Utility
    @test ndims(ad) == 1
    @test eltype(ad) == Float32

    free(ad)
end


@testset "compilation & execution" begin
    @compile dev reference_dummy """
    __global__ void reference_dummy()
    {
    }
    """

    cudacall(reference_dummy(), 1, 1, ())
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

        cudacall(reference_copy(), len, 1, (Ptr{Cfloat},Ptr{Cfloat}), input_dev, output_dev)
        output = Array(output_dev)
        @test_approx_eq input output

        free(input_dev)
        free(output_dev)
    end
end


destroy(ctx)

end
