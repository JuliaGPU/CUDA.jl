@testset "execution" begin

############################################################################################

@eval exec_dummy() = return nothing

@testset "cufunction" begin
    @test_throws UndefVarError cufunction(exec_undefined_kernel, ())

    cufunction(dev, exec_dummy, ())
    cufunction(dev, exec_dummy, Tuple{})

    # NOTE: other cases are going to be covered by tests below,
    #       as @cuda internally uses cufunction
end

############################################################################################


@testset "@cuda" begin

@test_throws UndefVarError @cuda dev (1,1) exec_undefined_kernel()


@testset "dimensions" begin
    @cuda dev (1,1) exec_dummy()
    @test_throws ArgumentError @cuda dev (0,0) exec_dummy()
end


@testset "shared memory" begin
    @cuda dev (1,1,1) exec_dummy()
end


@testset "streams" begin
    s = CuStream()
    @cuda dev (1,1,1,s) exec_dummy()
    destroy(s)
end


@testset "external kernels" begin
    @eval module KernelModule
        export exec_external_dummy
        exec_external_dummy() = return nothing
    end
    @cuda dev (1,1) KernelModule.exec_external_dummy()
    @eval begin
        using KernelModule
        @cuda dev (1,1) exec_external_dummy()
    end
end


@testset "return values" begin
    @eval exec_return_int_inner() = return 1
    @test_throws ErrorException @cuda dev (1,1) exec_return_int_inner()

    @eval function exec_return_int_outer()
        exec_return_int_inner()
        return nothing
    end
    @cuda dev (1,1) exec_return_int_outer()
end


@testset "calling device function" begin
    @eval @noinline exec_devfun_child() = return nothing
    @eval exec_devfun_parent() = exec_devfun_child()

    @cuda dev (1,1) exec_devfun_parent()
end

end


############################################################################################

@testset "argument passing" begin

dims = (16, 16)
len = prod(dims)

@testset "manually allocated" begin
    @eval function exec_pass_ptr(input, output)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x

        val = unsafe_load(input, i)
        unsafe_store!(output, val, i)

        return nothing
    end

    input = round.(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    @cuda dev (1,len) exec_pass_ptr(input_dev.ptr, output_dev.ptr)
    output = Array(output_dev)
    @test input ≈ output

    free(input_dev)
    free(output_dev)
end


@testset "scalar through single-value array" begin
    @eval function exec_pass_scalar(a, x)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        max = gridDim().x * blockDim().x
        if i == max
            val = unsafe_load(a, i)
            unsafe_store!(x, val)
        end

        return nothing
    end

    arr = round.(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda dev (1,len) exec_pass_scalar(arr_dev.ptr, val_dev.ptr)
    @test arr[dims...] ≈ Array(val_dev)[1]

    free(val_dev)
    free(arr_dev)
end


@testset "scalar through single-value array, using device function" begin
    @eval @noinline function exec_pass_scalar_devfun(a, x)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        max = gridDim().x * blockDim().x
        if i == max
            val = exec_pass_scalar_devfun_child(a, i)
            unsafe_store!(x, val)
        end

        return nothing
    end
    @eval function exec_pass_scalar_devfun_child(a, i)
        return unsafe_load(a, i)
    end

    arr = round.(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda dev (1,len) exec_pass_scalar_devfun(arr_dev.ptr, val_dev.ptr)
    @test arr[dims...] ≈ Array(val_dev)[1]

    free(val_dev)
    free(arr_dev)
end


@testset "ghost function parameters" begin
    # bug: ghost type function parameters are elided by the compiler

    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(Float32, len)

    @eval immutable ExecGhost end

    @eval function exec_pass_ghost(ghost, a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(c, unsafe_load(a,i)+unsafe_load(b,i), i)

        return nothing
    end
    @cuda dev (1,len) exec_pass_ghost(ExecGhost(), d_a.ptr, d_b.ptr, d_c.ptr)

    c = Array(d_c)
    @test a+b == c

    free(d_c)
    free(d_b)
    free(d_a)
end


@testset "tuples" begin
    # issue #7: tuples not passed by pointer

    @eval function exec_pass_tuples(keeps, out)
        if keeps[1]
            unsafe_store!(out, 1)
        else
            unsafe_store!(out, 2)
        end
        nothing
    end

    keeps = (true,)
    d_out = CuArray(Int, 1)

    @cuda dev (1,1) exec_pass_tuples(keeps, d_out.ptr)
    @test Array(d_out) == [1]

    free(d_out)
end


@testset "immutables" begin
    # issue #15: immutables not passed by pointer

    @eval function exec_pass_immutables(A, b)
        unsafe_store!(A, imag(b))
        nothing
    end

    A = CuArray(zeros(Float32, (1,)));
    x = Complex64(2,2)

    @cuda dev (1, 1) exec_pass_immutables(A.ptr, x)
    @test Array(A) == Float32[imag(x)]

    free(A)
end

end

############################################################################################

end
