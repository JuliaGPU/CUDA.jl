dev = CuDevice(0)
ctx = CuContext(dev)


## intrinsics

let
    buf = CuArray(Float32, 1)

    @target ptx function kernel_log10(a::CuDeviceArray{Float32}, i::Float32)
        a[1] = CUDAnative.log10(i)
        return nothing
    end

    @cuda (1, 1) kernel_log10(buf, Float32(100))
    val = Array(buf)
    @test val[1] ≈ 2.0

    free(buf)
end


## shared memory

# dynamic shmem

n = 1024
types = [Int32, Int64, Float32, Float64]

@target ptx function kernel_dynmem_typed(d::CuDeviceArray{Float32}, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuSharedMem(Float32)
    s[t] = d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

let
    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda (1, n, n*sizeof(Float32)) kernel_dynmem_typed(d_a, n)

    @assert reverse(a) == Array(d_a)
end

@target ptx function kernel_dynmem_typevar{T}(d::CuDeviceArray{T}, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuSharedMem(T)
    s[t] = d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

for T in types
    a = rand(T, n)
    d_a = CuArray(a)

    @cuda (1, n, n*sizeof(T)) kernel_dynmem_typevar(d_a, n)

    @assert reverse(a) == Array(d_a)
end


# static shmem

@target ptx function kernel_statmem_typed(d::CuDeviceArray{Float32}, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuSharedMem(Float32, 1024)
    s[t] = d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

let
    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda (1, n) kernel_statmem_typed(d_a, n)

    @assert reverse(a) == Array(d_a)
end

@target ptx function kernel_statmem_typevar{T}(d::CuDeviceArray{T}, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuSharedMem(T, 1024)
    s[t] = d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

for T in types
    a = rand(T, n)
    d_a = CuArray(a)

    @cuda (1, n) kernel_statmem_typevar(d_a, n)

    @assert reverse(a) == Array(d_a)
end


destroy(ctx)
