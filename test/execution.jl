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

@test_throws UndefVarError @cuda (1,1) exec_undefined_kernel()


@testset "reflection" begin
    @cuda (1,1) exec_dummy()

    @grab_output CUDAnative.@code_llvm @cuda (1,1) exec_dummy()
    @grab_output CUDAnative.@code_ptx @cuda (1,1) exec_dummy()
    @grab_output CUDAnative.@code_sass @cuda (1,1) exec_dummy()

    @grab_output CUDAnative.@code_llvm exec_dummy()
    @grab_output CUDAnative.@code_ptx exec_dummy()
    @grab_output CUDAnative.@code_sass exec_dummy()
end


@testset "reflection with argument conversion" begin
    @eval exec_dummy_array(a, i) = (a[1] = i; return nothing)

    a = CuArray{Float32}(1)

    @grab_output CUDAnative.@code_llvm @cuda (1,1) exec_dummy_array(a, 1)
    @grab_output CUDAnative.@code_ptx @cuda (1,1) exec_dummy_array(a, 1)
    @grab_output CUDAnative.@code_sass @cuda (1,1) exec_dummy_array(a, 1)

    @grab_output CUDAnative.@code_llvm exec_dummy_array(a, 1)
    @grab_output CUDAnative.@code_ptx exec_dummy_array(a, 1)
    @grab_output CUDAnative.@code_sass exec_dummy_array(a, 1)
end


@testset "dimensions" begin
    @cuda (1,1) exec_dummy()
    @test_throws ArgumentError @cuda (0,0) exec_dummy()
end


@testset "shared memory" begin
    @cuda (1,1,1) exec_dummy()
end


@testset "streams" begin
    s = CuStream()
    @cuda (1,1,1,s) exec_dummy()
end


@testset "external kernels" begin
    @eval module KernelModule
        export exec_external_dummy
        exec_external_dummy() = return nothing
    end
    @cuda (1,1) KernelModule.exec_external_dummy()
    @eval begin
        using KernelModule
        @cuda (1,1) exec_external_dummy()
    end

    @eval module WrapperModule
        using CUDAnative
        @eval exec_dummy() = return nothing
        wrapper() = @cuda (1,1) exec_dummy()
    end
    WrapperModule.wrapper()
end


@testset "return values" begin
    @eval exec_return_int_inner() = return 1
    @test_throws ErrorException @cuda (1,1) exec_return_int_inner()

    @eval function exec_return_int_outer()
        exec_return_int_inner()
        return nothing
    end
    @cuda (1,1) exec_return_int_outer()
end


@testset "calling device function" begin
    @eval @noinline exec_devfun_child(i) = return i+1
    @eval exec_devfun_parent() = (exec_devfun_child(1); return nothing)

    @cuda (1,1) exec_devfun_parent()
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
    output_dev = similar(input_dev)

    @cuda (1,len) exec_pass_ptr(input_dev.devptr, output_dev.devptr)
    output = Array(output_dev)
    @test input ≈ output
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

    @cuda (1,len) exec_pass_scalar(arr_dev.devptr, val_dev.devptr)
    @test arr[dims...] ≈ Array(val_dev)[1]
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

    @cuda (1,len) exec_pass_scalar_devfun(arr_dev.devptr, val_dev.devptr)
    @test arr[dims...] ≈ Array(val_dev)[1]
end


@testset "ghost function parameters" begin
    # bug: ghost type function parameters are elided by the compiler

    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = similar(d_a)

    @eval struct ExecGhost end

    @eval function exec_pass_ghost(ghost, a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(c, unsafe_load(a,i)+unsafe_load(b,i), i)

        return nothing
    end
    @cuda (1,len) exec_pass_ghost(ExecGhost(), d_a.devptr, d_b.devptr, d_c.devptr)

    c = Array(d_c)
    @test a+b == c
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
    d_out = CuArray{Int}(1)

    @cuda (1,1) exec_pass_tuples(keeps, d_out.devptr)
    @test Array(d_out) == [1]
end


@testset "immutables" begin
    # issue #15: immutables not passed by pointer

    @eval function exec_pass_immutables(A, b)
        unsafe_store!(A, imag(b))
        nothing
    end

    A = CuArray(zeros(Float32, (1,)))
    x = Complex64(2,2)

    @cuda (1, 1) exec_pass_immutables(A.devptr, x)
    @test Array(A) == Float32[imag(x)]
end

end

############################################################################################

end
