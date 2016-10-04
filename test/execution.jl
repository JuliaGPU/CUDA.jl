do_nothing() = return nothing

#
# cufunction
#

@test_throws UndefVarError cufunction(undefined_kernel, ())

cufunction(dev, do_nothing, ())
cufunction(dev, do_nothing, Tuple{})

# NOTE: other cases are going to be covered by tests below,
#       as @cuda internally uses cufunction



#
# @cuda
#

@test_throws UndefVarError @cuda dev (1,1) undefined_kernel()

# kernel dims
@cuda dev (1,1) do_nothing()
@test_throws ArgumentError @cuda dev (0,0) do_nothing()

# shared memory
@cuda dev (1,1,1) do_nothing()

# streams
s = CuStream()
@cuda dev (1,1,1,s) do_nothing()
destroy(s)

# external kernel
module KernelModule
    export do_more_nothing
    do_more_nothing() = return nothing
end
@cuda dev (1,1) KernelModule.do_more_nothing()
@eval begin
    using KernelModule
    @cuda dev (1,1) do_more_nothing()
end


## return values

retint() = return 1
@test_throws ErrorException @cuda dev (1,1) retint()

function call_retint()
    retint()
    return nothing
end
@cuda dev (1,1) call_retint()


## argument passing

dims = (16, 16)
len = prod(dims)

function ptr_copy(input, output)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    val = unsafe_load(input, i)
    unsafe_store!(output, val, i)

    return nothing
end

# manually allocated
let
    input = round.(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    @cuda dev (1,len) ptr_copy(input_dev.ptr, output_dev.ptr)
    output = Array(output_dev)
    @test input ≈ output

    free(input_dev)
    free(output_dev)
end

# scalar through single-value array
function ptr_lastvalue(a, x)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    max = gridDim().x * blockDim().x
    if i == max
        val = unsafe_load(a, i)
        unsafe_store!(x, val)
    end

    return nothing
end
let
    arr = round.(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda dev (1,len) ptr_lastvalue(arr_dev.ptr, val_dev.ptr)
    @test arr[dims...] ≈ Array(val_dev)[1]
end

# same, but using a device function
@noinline function ptr_lastvalue_devfun(a, x)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    max = gridDim().x * blockDim().x
    if i == max
        val = lastvalue_devfun(a, i)
        unsafe_store!(x, val)
    end

    return nothing
end
function lastvalue_devfun(a, i)
    return unsafe_load(a, i)
end
let
    arr = round.(rand(Float32, dims) * 100)
    val = Float32[0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda dev (1,len) ptr_lastvalue_devfun(arr_dev.ptr, val_dev.ptr)
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

    function map_inner(ghost, a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(c, unsafe_load(a,i)+unsafe_load(b,i), i)

        return nothing
    end
    @cuda dev (1,len) map_inner(Ghost(), d_a.ptr, d_b.ptr, d_c.ptr)

    c = Array(d_c)
    @test a+b == c
end

# issue #7: tuples not passed by pointer
let
    function kernel7(keeps, out)
        if keeps[1]
            unsafe_store!(out, 1)
        else
            unsafe_store!(out, 2)
        end
        nothing
    end

    keeps = (true,)
    d_out = CuArray(Int, 1)
    @cuda dev (1,1) kernel7(keeps, d_out.ptr)
    @test Array(d_out) == [1]
end

# issue #15: immutables not passed by pointer
let
    function kernel15(A, b)
        unsafe_store!(A, imag(b))
        nothing
    end

    A = CuArray(zeros(Float32, (1,)));
    x = Complex64(2,2)

    @cuda dev (1, 1) kernel15(A.ptr, x)

    @test Array(A) == Float32[imag(x)]
end

# issue: calling device function
let
    @noinline child() = return nothing
    parent() = child()

    @cuda dev (1,1) parent()
end