## math

let
    buf = CuArray(Float32, 1)

    @target ptx function kernel_log10(a, i)
        a[1] = CUDAnative.log10(i)
        return nothing
    end

    @cuda (1, 1) kernel_log10(buf, Float32(100))
    val = Array(buf)
    @test val[1] ≈ 2.0

    free(buf)
end


## shared memory

# smoke-test constructors
let
    # static
    @on_device @cuStaticSharedMem(Float32, 1)
    @on_device @cuStaticSharedMem(Float32, (1, 2))

    # dynamic
    @on_device @cuDynamicSharedMem(Float32, 1)
    @on_device @cuDynamicSharedMem(Float32, (1, 2))
    
    # dynamic with offset
    @on_device @cuDynamicSharedMem(Float32, 1, 8)
    @on_device @cuDynamicSharedMem(Float32, (1, 2), 8)
end

# dynamic shmem

n = 1024
types = [Int32, Int64, Float32, Float64]

@target ptx function shmem_dynamic_typed(d, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuDynamicSharedMem(Float32, n)
    s[t] = d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

let
    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda (1, n, n*sizeof(Float32)) shmem_dynamic_typed(d_a, n)

    @assert reverse(a) == Array(d_a)
end

@target ptx function shmem_dynamic_typevar{T}(d::CuDeviceArray{T}, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuDynamicSharedMem(T, n)
    s[t] = d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

for T in types
    a = rand(T, n)
    d_a = CuArray(a)

    @cuda (1, n, n*sizeof(T)) shmem_dynamic_typevar(d_a, n)

    @assert reverse(a) == Array(d_a)
end


# static shmem

@target ptx function shmem_static_typed(d, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuStaticSharedMem(Float32, 1024)
    s2 = @cuStaticSharedMem(Float32, 1024)  # catch aliasing

    s[t] = d[t]
    s2[t] = 2*d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

let
    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda (1, n) shmem_static_typed(d_a, n)

    @assert reverse(a) == Array(d_a)
end

@target ptx function shmem_static_typevar{T}(d::CuDeviceArray{T}, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuStaticSharedMem(T, 1024)
    s2 = @cuStaticSharedMem(T, 1024)  # catch aliasing

    s[t] = d[t]
    s2[t] = d[t]
    sync_threads()
    d[t] = s[tr]

    return nothing
end

for T in types
    a = rand(T, n)
    d_a = CuArray(a)

    @cuda (1, n) shmem_static_typevar(d_a, n)

    @assert reverse(a) == Array(d_a)
end


# dynamic shmem consisting of multiple arrays

# common use case 1: dynamic shmem consists of multiple homogeneous arrays
#                    -> split using `view`
@target ptx function shmem_dynamic_multi_homogeneous(a, b, n)
    t = threadIdx().x
    tr = n-t+1

    s = @cuDynamicSharedMem(eltype(a), 2*n)

    sa = view(s, 1:n)
    sa[t] = a[t]
    sync_threads()
    a[t] = sa[tr]

    sb = view(s, n+1:2*n)
    sb[t] = b[t]
    sync_threads()
    b[t] = sb[tr]

    return nothing
end

let
    a = rand(Float32, n)
    d_a = CuArray(a)

    b = rand(Float32, n)
    d_b = CuArray(b)

    @cuda (1, n, 2*n*sizeof(Float32)) shmem_dynamic_multi_homogeneous(d_a, d_b, n)

    @assert reverse(a) == Array(d_a)
    @assert reverse(b) == Array(d_b)
end

# common use case 2: dynamic shmem consists of multiple heterogeneous arrays
#                    -> construct using pointer offset
@target ptx function shmem_dynamic_multi_heterogeneous(a, b, n)
    t = threadIdx().x
    tr = n-t+1

    sa = @cuDynamicSharedMem(eltype(a), n)
    sa[t] = a[t]
    sync_threads()
    a[t] = sa[tr]

    sb = @cuDynamicSharedMem(eltype(b), n, n*sizeof(eltype(a)))
    sb[t] = b[t]
    sync_threads()
    b[t] = sb[tr]

    return nothing
end

let
    a = rand(Float32, n)
    d_a = CuArray(a)

    b = rand(Int64, n)
    d_b = CuArray(b)

    @cuda (1, n, n*sizeof(Float32) + n*sizeof(Int64)) shmem_dynamic_multi_heterogeneous(d_a, d_b, n)

    @assert reverse(a) == Array(d_a)
    @assert reverse(b) == Array(d_b)
end


## shuffle

n = 14

@target ptx function kernel_shuffle_down{T}(d::CuDeviceArray{T}, n)
    t = threadIdx().x
    if t <= n
        d[t] += shfl_down(d[t], n÷2)
    end
    return nothing
end

for T in types
    a = T[i for i in 1:n]
    d_a = CuArray(a)

    @cuda (1, nearest_warpsize(n)) kernel_shuffle_down(d_a, n)

    a[1:n÷2] += a[n÷2+1:end]
    @assert a == Array(d_a)
end


## parallel synchronization and communication

# voting

let
    d_a = CuArray(UInt32, 1)

    @target ptx function kernel_ballot(a, i)
        vote = vote_ballot(threadIdx().x == i)
        if threadIdx().x == 1
            a[1] = vote
        end

        return nothing
    end

    len = 4
    for i in 1:len
        @cuda (1,len) kernel_ballot(d_a, i)
        @test Array(d_a) == [2^(i-1)]
    end

    free(d_a)
end

let
    d_a = CuArray(UInt32, 1)

    @target ptx function kernel_any(a, i)
        vote = vote_any(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end

        return nothing
    end

    @cuda (1,2) kernel_any(d_a, 1)
    @test Array(d_a) == [1]

    @cuda (1,2) kernel_any(d_a, 2)
    @test Array(d_a) == [1]

    @cuda (1,2) kernel_any(d_a, 3)
    @test Array(d_a) == [0]

    free(d_a)
end

let
    d_a = CuArray(UInt32, 1)

    @target ptx function kernel_all(a, i)
        vote = vote_all(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end

        return nothing
    end

    @cuda (1,2) kernel_all(d_a, 1)
    @test Array(d_a) == [1]

    @cuda (1,2) kernel_all(d_a, 2)
    @test Array(d_a) == [0]

    @cuda (1,2) kernel_all(d_a, 3)
    @test Array(d_a) == [0]

    free(d_a)
end