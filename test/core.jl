@testset "core" begin

dev = CuDevice(0)
ctx = CuContext(dev)


@testset "API call wrapper" begin
    @cucall(:cuDriverGetVersion, (Ptr{Cint},), Ref{Cint}())

    @test_throws ErrorException @cucall(:nonExisting, ())
    CUDAdrv.trace(prefix=" ")

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

# Copy non-bit array
@test_throws ArgumentError begin
    # Something that's certainly not a bit type
    f =  x -> x*x
    input = [f for i=1:10]
    cu_input = CuArray(input)
end

# CuArray with not-bit elements
let
    @test_throws ArgumentError CuArray(Function, 10)
    @test_throws ArgumentError CuArray(Function, (10, 10))
end

# cu mem tests
let
    @test_throws ArgumentError cualloc(Function, 10)

    dev_array = CuArray(Int32, 10)
    cumemset(dev_array.ptr, UInt32(0), 10)
    host_array = Array(dev_array)

    for i in host_array
        @assert i == 0 "Memset failed on element $i"
    end

    free(dev_array)
end

destroy(ctx)

end
