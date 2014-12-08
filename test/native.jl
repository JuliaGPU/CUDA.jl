################################################################################
# Initialization
#

# IDEA: no core and native, but perf testing with for every native kernel a
#       c++ counterpart, submitting as a different executable

using CUDA, Base.Test

include("perfutil.jl")


#
# Set-up
#

dev = CuDevice(0)
ctx = CuContext(dev)

dims = (3, 4)
len = prod(dims)

cgctx = CuCodegenContext(ctx, dev)
ENV["ARCH"] = cgctx.arch
include("kernels/load.jl")



################################################################################
# Smoke testing
#

#
# ptx loading
#

# TODO: make this a dummy kernel and put vadd below

a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(Float32, dims)

launch(reference_vadd(), len, 1, (a_dev.ptr, b_dev.ptr, c_dev.ptr))
c = to_host(c_dev)

free(a_dev)
free(b_dev)
free(c_dev)

@test_approx_eq (a + b) c


#
# @cuda macro
#

# kernel dims
@target ptx kernel_empty() = return nothing
@test_throws ErrorException @eval begin
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

# unlowered call
if PERFORMANCE
    i = 0
    @timeit begin # setup
            i += 1
            fname = symbol("kernel_dummy_$i")
            @eval @target ptx $fname() = return nothing
        end begin # benchmark
            @eval @cuda (1, 1) $fname()
        end begin # verification
        end begin # teardown
        end "macro_full" "unlowered call to @cuda (staged function execution, kernel compilation, runtime API interactions and asynchronous kernel launch)"

    synchronize(ctx)
end


# pre-lowered call
if PERFORMANCE
    @target ptx kernel_dummy() = return nothing

    @timeit begin # setup
        end begin # benchmark
            @cuda (1, 1) kernel_dummy()
        end begin # verification
        end begin # teardown
        end "macro_lowered" "lowered call to @cuda (runtime API interactions and asynchronous kernel launch)"

    synchronize(ctx)
end



################################################################################
# Argument passing
#

# TODO: new context with CTX_SCHED_BLOCKING_SYNC flag instead of synchronize(ctx)

#
# manually managed data
#

a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(Float32, dims)

@cuda (len, 1) kernel_vadd(a_dev, b_dev, c_dev)
c = to_host(c_dev)
@test_approx_eq (a + b) c

free(a_dev)
free(b_dev)
free(c_dev)


#
# auto-managed host data
#

a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)
c = Array(Float32, dims)

@cuda (len, 1) kernel_vadd(CuIn(a), CuIn(b), CuOut(c))
@test_approx_eq (a + b) c


#
# auto-managed host data, without specifying type
#

a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)
c = Array(Float32, dims)

@cuda (len, 1) kernel_vadd(a, b, c)
@test_approx_eq (a + b) c


#
# auto-managed host data, without specifying type, not using containers
#

a = rand(Float32, dims)
b = rand(Float32, dims)
c = Array(Float32, dims)

@cuda (len, 1) kernel_vadd(round(a*100), round(b*100), c)
@test_approx_eq (round(a*100) + round(b*100)) c


#
# scalar through single-value array
#

a = round(rand(Float32, dims) * 100)
x = Float32[0]

@cuda (len, 1) kernel_lastvalue(CuIn(a), CuOut(x))
@test_approx_eq a[dims...] x[1]


################################################################################
# Performance tests
#

#
# Argument passing
#

# TODO: PERFORMANCE==0 just runs single iteration
if PERFORMANCE
    @timeit begin # setup
            input = round(rand(Float32, dims) * 100)
        end begin # benchmark
            input_dev = CuArray(input)
            output_dev = CuArray(Float32, dims)

            launch(reference_copy(), len, 1, (input_dev.ptr, output_dev.ptr))
            output = to_host(output_dev)

            free(input_dev)
            free(output_dev)
        end begin # verification
            @test_approx_eq input output
        end begin # teardown
        end "copy_reference" "vector copy reference execution"

    @timeit begin # setup
            input = round(rand(Float32, dims) * 100)
        end begin # benchmark
            input_dev = CuArray(input)
            output_dev = CuArray(Float32, dims)

            @cuda (len, 1) kernel_copy(input_dev, output_dev)
            output = to_host(output_dev)

            free(input_dev)
            free(output_dev)
        end begin # verification
            @test_approx_eq input output
        end begin # teardown
        end "copy_manual" "vector copy on manually allocated GPU arrays"

    @timeit begin # setup
            input = round(rand(Float32, dims) * 100)
            output = Array(Float32, dims)
        end begin # benchmark
            @cuda (len, 1) kernel_copy(CuIn(input), CuOut(output))
        end begin # verification
            @test_approx_eq input output
        end begin # teardown
        end "copy_managed" "vector copy on managed GPU arrays"
end


destroy(cgctx)
