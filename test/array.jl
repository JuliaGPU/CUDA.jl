@testset "array" begin

let
    # inner constructors
    let
        arr = CuArray{Int,1}((2,))
        buf = arr.buf
        CuArray{Int,1}((2,), buf)
    end

    # outer constructors
    for I in [Int32,Int64]
        a = I(1)
        b = I(2)

        # partially parameterized
        CuArray{I}(b)
        CuArray{I}((b,))
        CuArray{I}(a,b)
        CuArray{I}((a,b))

        # fully parameterized
        CuArray{I,1}(b)
        CuArray{I,1}((b,))
        @test_throws MethodError CuArray{I,1}(a,b)
        @test_throws MethodError CuArray{I,1}((a,b))
        @test_throws MethodError CuArray{I,2}(b)
        @test_throws MethodError CuArray{I,2}((b,))
        CuArray{I,2}(a,b)
        CuArray{I,2}((a,b))
    end

    # type aliases
    let
        CuVector{Int}(1)
        CuMatrix{Int}(1,1)
    end

    # similar
    let a = CuArray{Int}(2)
        similar(a)
        similar(a, Float32)
    end
    let a = CuArray{Int}((1,2))
        similar(a)
        similar(a, Float32)
        similar(a, 2)
        similar(a, (2,1))
        similar(a, Float32, 2)
        similar(a, Float32, (2,1))
    end

    # comparisons
    let a = CuArray{Int}(2)
        @test a == a
        @test a != CuArray{Int}(2)
        @test a != CuArray{Int}(3)
    end

    # conversions
    let
        buf = CUDAdrv.Buffer(C_NULL, 0, CuContext(C_NULL))
        a = CuArray{Int,1}((1,), buf)

        @test Base.unsafe_convert(Ptr{Int}, Base.cconvert(Ptr{Int}, a)) == C_NULL
    end

    # copy: size mismatches
    let
        a = rand(Float32, 10)
        ad = CuArray{Float32}(5)
        bd = CuArray{Float32}(10)

        @test_throws ArgumentError copy!(ad, a)
        @test_throws ArgumentError copy!(a, ad)
        @test_throws ArgumentError copy!(ad, bd)
    end

    # copy to and from device
    let
        cpu = rand(Float32, 10)
        gpu = CuArray{Float32}(10)

        copy!(gpu, cpu)

        cpu_back = Array{Float32}(10)
        copy!(cpu_back, gpu)
        @assert cpu == cpu_back
    end

    # same, but with convenience functions
    let
        cpu = rand(Float32, 10)

        gpu = CuArray(cpu)
        cpu_back = Array(gpu)

        @assert cpu == cpu_back
    end

    # copy on device
    let gpu = CuArray(rand(Float32, 10))
        gpu_copy = copy(gpu)
        @test gpu != gpu_copy
        @test Array(gpu) == Array(gpu_copy)
    end

    # utility
    let gpu = CuArray{Float32}(5)
        @test ndims(gpu) == 1
        @test size(gpu, 1) == 5
        @test size(gpu, 2) == 1
        @test eltype(gpu) == Float32
        @test eltype(typeof(gpu)) == Float32
        @test sizeof(gpu) == 5*sizeof(Float32)
    end

    # printing
    let gpu = CuArray([42])
        show(DevNull, gpu)
        show(DevNull, "text/plain", gpu)
    end

    # finalizers
    let gpu = CuArray([42])
        finalize(gpu) # triggers early finalization
        finalize(gpu) # shouldn't re-run the finalizer
    end
end

let
    # ghost type
    @test_throws ArgumentError CuArray([x->x*x for i=1:10])

    # non-isbits elements
    @test_throws ArgumentError CuArray(["foobar" for i=1:10])
    @test_throws ArgumentError CuArray{Function}(10)
    @test_throws ArgumentError CuArray{Function}((10, 10))
end

end