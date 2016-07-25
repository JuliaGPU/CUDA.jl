@target ptx do_nothing() = return nothing

@test_throws UndefVarError @cuda (1, 1) undefined_kernel()

# kernel dims
@cuda (1, 1) do_nothing()
@test_throws ArgumentError @cuda (0, 0) do_nothing()

# external kernel
module KernelModule
    export do_more_nothing
    @target ptx do_more_nothing() = return nothing
end
@cuda (1, 1) KernelModule.do_more_nothing()
@eval begin
    using KernelModule
    @cuda (1, 1) do_more_nothing()
end


## argument passing

dims = (16, 16)
len = prod(dims)

@target ptx function ptr_copy(input, output)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    val = unsafe_load(input, i)
    unsafe_store!(output, val, i)

    return nothing
end

# manually allocated
let
    input = round(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    @cuda (1,len) ptr_copy(input_dev.ptr, output_dev.ptr)
    output = Array(output_dev)
    @test input ≈ output

    free(input_dev)
    free(output_dev)
end

# scalar through single-value array
@target ptx function ptr_lastvalue(a, x)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    max = gridDim().x * blockDim().x
    if i == max
        val = unsafe_load(a, i)
        unsafe_store!(x, val)
    end

    return nothing
end
let
    arr = round(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda (1,len) ptr_lastvalue(arr_dev.ptr, val_dev.ptr)
    @test arr[dims...] ≈ Array(val_dev)[1]
end

# same, but using a device function
@target ptx @noinline function ptr_lastvalue_devfun(a, x)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    max = gridDim().x * blockDim().x
    if i == max
        val = lastvalue_devfun(a, i)
        unsafe_store!(x, val)
    end

    return nothing
end
@target ptx function lastvalue_devfun(a, i)
    return unsafe_load(a, i)
end
let
    arr = round(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda (1,len) ptr_lastvalue_devfun(arr_dev.ptr, val_dev.ptr)
    @test arr[dims...] ≈ Array(val_dev)[1]
end

# bug: ghost type function parameters are elided by the compiler
let
    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(Float32, len)

    immutable Ghost end

    @target ptx function map_inner(ghost, a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(c, unsafe_load(a,i)+unsafe_load(b,i), i)

        return nothing
    end
    @cuda (1,len) map_inner(Ghost(), d_a.ptr, d_b.ptr, d_c.ptr)

    c = Array(d_c)
    @test a+b == c
end
