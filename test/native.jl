################################################################################
# Initialization
#

using CUDA, Base.Test

include("perfutil.jl")


#
# Set-up
#

dev = CuDevice(0)
ctx = CuContext(dev)

cgctx = CuCodegenContext(ctx, dev)
ENV["ARCH"] = cgctx.arch
include("kernels/load.jl")



################################################################################
# Smoke testing
#

#
# ptx loading
#

launch(reference_dummy(), 1, 1, ())


#
# @cuda macro
#

@cuda (1, 1) kernel_dummy()

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

# pre-lowered call
@target ptx kernel_dummy() = return nothing
@timeit begin # setup
    end begin # benchmark
        @cuda (1, 1) kernel_dummy()
    end begin # verification
    end begin # teardown
    end "macro_lowered" "lowered call to @cuda (runtime API interactions and asynchronous kernel launch)"
synchronize(ctx)


#
# Argument passing
#

dims = (512, 512)
len = prod(dims)

# manual allocation
@timeit begin # setup
        input = round(rand(Float32, dims) * 100)
    end begin # benchmark
        input_dev = CuArray(input)
        output_dev = CuArray(Float32, dims)

        CUDA.launch(reference_copy(), len, 1, (input_dev.ptr, output_dev.ptr))
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

# auto-managed host data
@timeit begin # setup
        input = round(rand(Float32, dims) * 100)
        output = Array(Float32, dims)
    end begin # benchmark
        @cuda (len, 1) kernel_copy(CuIn(input), CuOut(output))
    end begin # verification
        @test_approx_eq input output
    end begin # teardown
    end "copy_managed" "vector copy on managed GPU arrays"

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

# scalar through single-value array
let
    arr = round(rand(Float32, dims) * 100)
    val = Float32[0]

    @cuda (len, 1) kernel_lastvalue(CuIn(arr), CuOut(val))
    @test_approx_eq arr[dims...] val[1]
end


destroy(cgctx)
