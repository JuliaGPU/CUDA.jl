dev = CuDevice(0)
ctx = CuContext(dev)
cgctx = CuCodegenContext(ctx, dev)


## @cuda macro

@target ptx kernel_empty() = return nothing

# kernel dims
@test_throws AssertionError @eval begin
    @cuda (0, 0) kernel_empty()
end
@eval begin
    @cuda (1, 1) kernel_empty()
end

# kernel name
@test_throws ErrorException @eval begin
    @cuda (1, 1) Module.kernel_foo()
end
@test_throws ErrorException @eval begin
    @cuda (1, 1) InvalidPrefixedKernel()
end

# external kernel
module KernelModule
    export kernel_empty2
    @target ptx kernel_empty2() = return nothing
end
@eval begin
    using KernelModule
    @cuda (1, 1) kernel_empty2()
end

# TODO: test device function call with @noinline


## Return values

# TODO
# TODO: test the actual error message
# @test_throws ErrorException @eval begin
#     @target ptx kernel_retint() = return 1
#     @cuda (1, 1) kernel_retint()
# end


## Argument passing

dims = (16, 16)
len = prod(dims)

@target ptx function kernel_copy(input::CuDeviceArray{Float32},
                                 output::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    output[i] = input[i]

    return nothing
end

# manual allocation
let
    input = round(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    @cuda (len, 1) kernel_copy(input_dev, output_dev)
    output = to_host(output_dev)
    @test_approx_eq input output

    free(input_dev)
    free(output_dev)
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
    @test_throws ArgumentError CUDA.cualloc(Function, 10)

    dev_array = CuArray(Int32, 10)
    CUDA.cumemset(dev_array.ptr, UInt32(0), 10)
    host_array = to_host(dev_array)

    for i in host_array
        @assert i == 0 "Memset failed on element $i"
    end

    CUDA.free(dev_array)

end

# auto-managed host data
let
    input = round(rand(Float32, dims) * 100)
    output = Array(Float32, dims)

    @cuda (len, 1) kernel_copy(CuIn(input), CuOut(output))
    @test_approx_eq input output
end

# auto-managed host data, without specifying type
let
    input = round(rand(Float32, dims) * 100)
    output = Array(Float32, dims)

    @cuda (len, 1) kernel_copy(input, output)
    @test_approx_eq input output
end

# auto-managed host data, without specifying type, not using containers
let
    input = rand(Float32, dims)
    output = Array(Float32, dims)

    @cuda (len, 1) kernel_copy(round(input*100), output)
    @test_approx_eq round(input*100) output
end

@target ptx function kernel_lastvalue(a::CuDeviceArray{Float32},
                                      x::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    max = gridDim().x * blockDim().x
    if i == max
        x[1] = a[i]
    end

    return nothing
end

# scalar through single-value array
let
    arr = round(rand(Float32, dims) * 100)
    val = Float32[0]

    @cuda (len, 1) kernel_lastvalue(CuIn(arr), CuOut(val))
    @test_approx_eq arr[dims...] val[1]
end

@target ptx function kernel_lastvalue_devfun(a::CuDeviceArray{Float32},
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

# same, but using a device function
let
    arr = round(rand(Float32, dims) * 100)
    val = Float32[0]

    @cuda (len, 1) kernel_lastvalue_devfun(CuIn(arr), CuOut(val))
    @test_approx_eq arr[dims...] val[1]
end


destroy(cgctx)
destroy(ctx)
