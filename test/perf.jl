using CUDA, Base.Test

include("perfutil.jl")

# TODO: deduplicate with runtests? only perf if arg, no perf.jl?


# set-up

dev = CuDevice(0)
ctx = CuContext(dev)

initialize_codegen(ctx, dev)


#
# @cuda
#

i = 0
@timeit_init begin
        @eval @cuda (0, 0) $fname()
    end begin
        # initialization
        i += 1
        fname = symbol("kernel_dummy_$i")
        @eval @target ptx $fname() = return nothing
    end "macro_full" "unlowered call to @cuda (staged function execution, kernel compilation, runtime API interactions and asynchronous kernel launch)"

synchronize(ctx)


@target ptx kernel_dummy() = return nothing

@timeit begin
        @cuda (0, 0) kernel_dummy()
    end "macro_lowered" "lowered call to @cuda (runtime API interactions and asynchronous kernel launch)"

synchronize(ctx)


#
# vadd
#

@target ptx function kernel_vadd(a::CuDeviceArray{Float32}, b::CuDeviceArray{Float32},
                                 c::CuDeviceArray{Float32})
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    c[i] = a[i] + b[i]

    return nothing
end

siz = (3, 4)
len = prod(siz)


# test 1: manually managed data

# TODO: new context with CTX_SCHED_BLOCKING_SYNC flag instead of synchronize(ctx)

a = round(rand(Float32, siz) * 100)
b = round(rand(Float32, siz) * 100)

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(Float32, siz)

@timeit begin
        @cuda (len, 1) kernel_vadd(a_dev, b_dev, c_dev)
        synchronize(ctx)
    end "vadd_manual" "vector addition on manually added GPU arrays"

c = to_host(c_dev)
@test_approx_eq (a + b) c

free(a_dev)
free(b_dev)
free(c_dev)
