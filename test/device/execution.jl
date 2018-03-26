@testset "execution" begin

############################################################################################

@eval exec_dummy() = return nothing

@testset "cufunction" begin
    @test_throws UndefVarError cufunction(exec_undefined_kernel, ())

    cufunction(dev, exec_dummy, Tuple{})

    # NOTE: other cases are going to be covered by tests below,
    #       as @cuda internally uses cufunction
end

############################################################################################


@testset "@cuda" begin

@test_throws UndefVarError @cuda exec_undefined_kernel()
@test_throws MethodError @cuda exec_dummy(1)


@testset "compilation params" begin
    @cuda exec_dummy()

    @test_throws CuError @cuda threads=2 maxthreads=1 exec_dummy()
    @cuda threads=2 exec_dummy()
end


@testset "reflection" begin
    CUDAnative.code_lowered(exec_dummy, Tuple{})
    CUDAnative.code_typed(exec_dummy, Tuple{})
    CUDAnative.code_warntype(devnull, exec_dummy, Tuple{})
    CUDAnative.code_llvm(devnull, exec_dummy, Tuple{})
    CUDAnative.code_ptx(devnull, exec_dummy, Tuple{})
    CUDAnative.code_sass(devnull, exec_dummy, Tuple{})

    @device_code_lowered @cuda exec_dummy()
    @device_code_typed @cuda exec_dummy()
    @device_code_warntype io=devnull @cuda exec_dummy()
    @device_code_llvm io=devnull @cuda exec_dummy()
    @device_code_ptx io=devnull @cuda exec_dummy()
    @device_code_sass io=devnull @cuda exec_dummy()

    @test_throws ErrorException @device_code_lowered nothing
end


@testset "shared memory" begin
    @cuda shmem=1 exec_dummy()
end


@testset "streams" begin
    s = CuStream()
    @cuda stream=s exec_dummy()
end


@testset "external kernels" begin
    @eval module KernelModule
        export exec_external_dummy
        exec_external_dummy() = return nothing
    end
    import ...KernelModule
    @cuda KernelModule.exec_external_dummy()
    @eval begin
        using ...KernelModule
        @cuda exec_external_dummy()
    end

    @eval module WrapperModule
        using CUDAnative
        @eval exec_dummy() = return nothing
        wrapper() = @cuda exec_dummy()
    end
    WrapperModule.wrapper()
end


@testset "return values" begin
    @eval exec_return_int_inner() = return 1
    @test_throws ArgumentError @cuda exec_return_int_inner()

    @eval function exec_return_int_outer()
        exec_return_int_inner()
        return nothing
    end
    @cuda exec_return_int_outer()
end


@testset "calling device function" begin
    @eval @noinline exec_devfun_child(i) = sink(i)
    @eval exec_devfun_parent() = (exec_devfun_child(1); return nothing)

    @cuda exec_devfun_parent()
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
    output = similar(input)

    input_dev = Mem.upload(input)
    output_dev = Mem.alloc(input)

    @cuda threads=len exec_pass_ptr(Base.unsafe_convert(Ptr{Float32}, input_dev),
                                    Base.unsafe_convert(Ptr{Float32}, output_dev))
    Mem.download!(output, output_dev)
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
    val = [0f0]

    arr_dev = Mem.upload(arr)
    val_dev = Mem.upload(val)

    @cuda threads=len exec_pass_scalar(Base.unsafe_convert(Ptr{Float32}, arr_dev),
                                       Base.unsafe_convert(Ptr{Float32}, val_dev))
    @test arr[dims...] ≈ Mem.download(eltype(val), val_dev)[1]
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
    val = [0f0]

    arr_dev = Mem.upload(arr)
    val_dev = Mem.upload(val)

    @cuda threads=len exec_pass_scalar_devfun(Base.unsafe_convert(Ptr{Float32}, arr_dev),
                                              Base.unsafe_convert(Ptr{Float32}, val_dev))
    @test arr[dims...] ≈ Mem.download(eltype(val), val_dev)[1]
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
    d_out = Mem.alloc(Int)

    @cuda exec_pass_tuples(keeps, Base.unsafe_convert(Ptr{Int}, d_out))
    @test Mem.download(Int, d_out) == [1]
end

@testset "tuple of arrays" begin
    @eval function exec_pass_tuples(xs)
        xs[1][1] = xs[2][1]
        nothing
    end

    x1 = CuTestArray([1])
    x2 = CuTestArray([2])

    @cuda exec_pass_tuples((x1, x2))
    @test Array(x1) == [2]
end


@testset "ghost function parameters" begin
    # bug: ghost type function parameters are elided by the compiler

    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)
    c = similar(a)

    d_a = Mem.upload(a)
    d_b = Mem.upload(b)
    d_c = Mem.alloc(c)

    @eval struct ExecGhost end

    @eval function exec_pass_ghost(ghost, a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(c, unsafe_load(a,i)+unsafe_load(b,i), i)

        return nothing
    end
    @cuda threads=len exec_pass_ghost(ExecGhost(),
                                      Base.unsafe_convert(Ptr{Float32}, d_a),
                                      Base.unsafe_convert(Ptr{Float32}, d_b),
                                      Base.unsafe_convert(Ptr{Float32}, d_c))
    Mem.download!(c, d_c)
    @test a+b == c


    # bug: ghost type function parameters confused aggregate type rewriting

    @eval function exec_pass_ghost_aggregate(ghost, out, aggregate)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(out, aggregate[1], i)

        return nothing
    end
    @cuda threads=len exec_pass_ghost_aggregate(ExecGhost(),
                                                Base.unsafe_convert(Ptr{Float32}, d_c),
                                                (42,))

    Mem.download!(c, d_c)
    @test all(val->val==42, c)
end


@testset "immutables" begin
    # issue #15: immutables not passed by pointer

    @eval function exec_pass_immutables(ptr, b)
        unsafe_store!(ptr, imag(b))
        return nothing
    end

    buf = Mem.upload([0f0])
    x = ComplexF32(2,2)

    @cuda exec_pass_immutables(Base.unsafe_convert(Ptr{Float32}, buf), x)
    @test Mem.download(Float32, buf) == [imag(x)]
end


@testset "automatic recompilation" begin
    buf = Mem.alloc(Int)

    @eval exec_265(ptr) = (unsafe_store!(ptr, 1); return nothing)

    @cuda exec_265(Base.unsafe_convert(Ptr{Int}, buf))
    @test Mem.download(Int, buf) == [1]

    @eval exec_265(ptr) = (unsafe_store!(ptr, 2); return nothing)

    @cuda exec_265(Base.unsafe_convert(Ptr{Int}, buf))
    @test Mem.download(Int, buf) == [2]
end


@testset "non-isbits arguments" begin
    @eval exec_pass_nonbits_unused(T, i) = (sink(i); return)
    @cuda exec_pass_nonbits_unused(Int, 1)

    @eval exec_pass_nonbits_specialized(T, i) = (sink(unsafe_trunc(T,i)); return)
    @cuda exec_pass_nonbits_specialized(Int, 1.)

    @eval exec_pass_nonbits_used(i) = (sink(unsafe_trunc(Int,i)); return)
    @test_throws ArgumentError @cuda exec_pass_nonbits_used(big"1")
end


@testset "splatting" begin
    @eval function exec_splat(out, a, b)
        unsafe_store!(out, a+b)
        return
    end

    out = [0]
    out_dev = Mem.upload(out)
    out_ptr = Base.unsafe_convert(Ptr{eltype(out)}, out_dev)

    @cuda exec_splat(out_ptr, 1, 2)
    @test Mem.download(eltype(out), out_dev)[1] == 3

    all_splat = (out_ptr, 3, 4)
    @cuda exec_splat(all_splat...)
    @test Mem.download(eltype(out), out_dev)[1] == 7

    partial_splat = (5, 6)
    @cuda exec_splat(out_ptr, partial_splat...)
    @test Mem.download(eltype(out), out_dev)[1] == 11
end

end

############################################################################################

@testset "profile" begin

CUDAnative.@profile @cuda exec_dummy()

end

############################################################################################

end
