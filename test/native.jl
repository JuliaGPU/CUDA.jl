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

initialize_codegen(ctx, dev)


# test 1: manually managed data

a = round(rand(Float32, siz) * 100)
b = round(rand(Float32, siz) * 100)

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(Float32, siz)

@cuda (len, 1) kernel_vadd(a_dev, b_dev, c_dev)
c = to_host(c_dev)
@test_approx_eq (a + b) c

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

@cuda (len, 1) GPUModule.vadd(round(a*100), round(b*100), c)
@test_approx_eq (round(a*100) + round(b*100)) c
