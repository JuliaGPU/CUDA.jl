#
# Initialization
#

using CUDA, Base.Test

# kernels

@target ptx function kernel_vadd(a::CuDeviceArray{Float32}, b::CuDeviceArray{Float32},
                                 c::CuDeviceArray{Float32})
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    c[i] = a[i] + b[i]

    return nothing
end

@target ptx function kernel_scalaradd(a::CuDeviceArray{Float32}, x)
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    a[i] = a[i] + x
end


# set-up

dev = CuDevice(0)
ctx = CuContext(dev)

siz = (3, 4)
len = prod(siz)

cgctx = CuCodegenContext(ctx, dev)

# TODO: these are 2D arrays -- why isn't CuArray 2D?


#
# @cuda macro
#

if PERFORMANCE
    i = 0
    @timeit_init begin
            @eval @cuda (1, 1) $fname()
        end begin
            # initialization
            i += 1
            fname = symbol("kernel_dummy_$i")
            @eval @target ptx $fname() = return nothing
        end "macro_full" "unlowered call to @cuda (staged function execution, kernel compilation, runtime API interactions and asynchronous kernel launch)"

    synchronize(ctx)
end


if PERFORMANCE
    @target ptx kernel_dummy() = return nothing

    @timeit begin
            @cuda (1, 1) kernel_dummy()
        end "macro_lowered" "lowered call to @cuda (runtime API interactions and asynchronous kernel launch)"

    synchronize(ctx)
end


#
# Data management
#

# TODO: new context with CTX_SCHED_BLOCKING_SYNC flag instead of synchronize(ctx)


# test 1: manually managed data

a = round(rand(Float32, siz) * 100)
b = round(rand(Float32, siz) * 100)

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(Float32, siz)

@cuda (len, 1) kernel_vadd(a_dev, b_dev, c_dev)
c = to_host(c_dev)
@test_approx_eq (a + b) c

if PERFORMANCE
    @timeit begin
            @cuda (len, 1) kernel_vadd(a_dev, b_dev, c_dev)
            synchronize(ctx)
        end "vadd_manual" "vector addition on manually added GPU arrays"
end

free(a_dev)
free(b_dev)
free(c_dev)


# test 2: auto-managed host data

a = round(rand(Float32, siz) * 100)
b = round(rand(Float32, siz) * 100)
c = Array(Float32, siz)

@cuda (len, 1) kernel_vadd(CuIn(a), CuIn(b), CuOut(c))
@test_approx_eq (a + b) c


# test 3: auto-managed host data, without specifying type

a = round(rand(Float32, siz) * 100)
b = round(rand(Float32, siz) * 100)
c = Array(Float32, siz)

@cuda (len, 1) kernel_vadd(a, b, c)
@test_approx_eq (a + b) c


# test 4: auto-managed host data, without specifying type, not using containers

a = rand(Float32, siz)
b = rand(Float32, siz)
c = Array(Float32, siz)

@cuda (len, 1) kernel_vadd(round(a*100), round(b*100), c)
@test_approx_eq (round(a*100) + round(b*100)) c


destroy(cgctx)
