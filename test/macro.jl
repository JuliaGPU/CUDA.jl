dev = CuDevice(0)
ctx = CuContext(dev)

@target ptx do_nothing() = return nothing

@test_throws UndefVarError @cuda (1, 1) undefined_kernel()

# kernel dims
@test_throws ArgumentError @cuda (0, 0) do_nothing()
@cuda (1, 1) do_nothing()

# external kernel
@cuda (1, 1) KernelModule.do_more_nothing()
@eval begin
    using KernelModule
    @cuda (1, 1) do_more_nothing()
end


## argument passing

dims = (16, 16)
len = prod(dims)

@target ptx function array_copy(input::CuDeviceArray{Float32},
                                output::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    output[i] = input[i]

    return nothing
end

# manually allocated
let
    input = round(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    @cuda (len, 1) array_copy(input_dev, output_dev)
    output = Array(output_dev)
    @test_approx_eq input output

    free(input_dev)
    free(output_dev)
end

# scalar through single-value array
@target ptx function array_lastvalue(a::CuDeviceArray{Float32},
                                     x::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    max = gridDim().x * blockDim().x
    if i == max
        x[1] = a[i]
    end

    return nothing
end
let
    arr = round(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda (len, 1) array_lastvalue(arr_dev, val_dev)
    @test_approx_eq arr[dims...] Array(val_dev)[1]
end

# same, but using a device function
# NOTE: disabled because of #15276 / #15967
@target ptx @noinline function array_lastvalue_devfun(a::CuDeviceArray{Float32},
                                                      x::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    max = gridDim().x * blockDim().x
    if i == max
        x[1] = lastvalue_devfun(a, i)
    end

    return nothing
end
@target ptx function lastvalue_devfun(a::CuDeviceArray{Float32}, i)
    return a[i]
end
let
    arr = round(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    # @cuda (len, 1) array_lastvalue_devfun(arr_dev, val_dev)
    # @test_approx_eq arr[dims...] Array(val_dev)[1]
end

# bug: first argument function parameters are elided by the compiler
let
    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(Float32, len)

    @target ptx add(a,b) = a+b
    @target ptx function map_inner{F}(fun::F, a, b, c)
        i = blockIdx().x + (threadIdx().x-1) * gridDim().x
        c[i] = fun(a[i], b[i])

        return nothing
    end
    # NOTE: the add isn't actually passed to the GPU,
    #       and functions at code_llvm etc only contain 3 arguments
    @cuda (len, 1) map_inner(add, d_a, d_b, d_c)

    c = Array(d_c)
    @test a+b == c
end

# bug: same with ghost types
let
    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(Float32, len)

    immutable Ghost end

    @target ptx function map_inner(ghost, a, b, c)
        i = blockIdx().x + (threadIdx().x-1) * gridDim().x
        c[i] = a[i] + b[i]

        return nothing
    end
    @cuda (len, 1) map_inner(Ghost(), d_a, d_b, d_c)

    c = Array(d_c)
    @test a+b == c
end



destroy(ctx)
