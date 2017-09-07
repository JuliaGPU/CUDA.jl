@testset "intrinsics" begin

############################################################################################

@testset "math" begin
    buf = CuArray{Float32}(1)

    @eval function kernel_math_log10(a, i)
        a[1] = CUDAnative.log10(i)
        return nothing
    end

    @cuda (1, 1) kernel_math_log10(buf, Float32(100))
    val = Array(buf)
    @test val[1] ≈ 2.0
end



############################################################################################

@testset "I/O" begin

@testset "printing" begin
    _, out = @grab_output @on_device @cuprintf("")
    @test out == ""

    _, out = @grab_output @on_device @cuprintf("Testing...\n")
    @test out == "Testing...\n"

    _, out = @grab_output @on_device @cuprintf("Testing %ld...\n", Int64(42))
    @test out == "Testing 42...\n"

    _, out = @grab_output @on_device @cuprintf("Testing %u %u...\n", blockIdx().x, threadIdx().x)
    @test out == "Testing 1 1...\n"

    _, out = @grab_output @on_device begin
        @cuprintf("foo")
        @cuprintf("bar\n")
    end
    @test out == "foobar\n"
end

end



############################################################################################

# a composite type to test for more complex element types
@eval struct RGB{T}
    r::T
    g::T
    b::T
end

@testset "shared memory" begin

n = 1024
types = [Int32, Int64, Float32, Float64]

@testset "constructors" begin
    # static
    @on_device @cuStaticSharedMem(Float32, 1)
    @on_device @cuStaticSharedMem(Float32, (1,2))
    @on_device @cuStaticSharedMem(Tuple{Float32, Float32}, 1)
    @on_device @cuStaticSharedMem(Tuple{Float32, Float32}, (1,2))
    @on_device @cuStaticSharedMem(Tuple{RGB{Float32}, UInt32}, 1)
    @on_device @cuStaticSharedMem(Tuple{RGB{Float32}, UInt32}, (1,2))

    # dynamic
    @on_device @cuDynamicSharedMem(Float32, 1)
    @on_device @cuDynamicSharedMem(Float32, (1, 2))
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, 1)
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, (1,2))
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, 1)
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, (1,2))

    # dynamic with offset
    @on_device @cuDynamicSharedMem(Float32, 1, 8)
    @on_device @cuDynamicSharedMem(Float32, (1,2), 8)
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, 1, 8)
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, (1,2), 8)
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, 1, 8)
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, (1,2), 8)
end


@testset "dynamic shmem" begin

@testset "statically typed" begin
    @eval function kernel_shmem_dynamic_typed(d, n)
        t = threadIdx().x
        tr = n-t+1

        s = @cuDynamicSharedMem(Float32, n)
        s[t] = d[t]
        sync_threads()
        d[t] = s[tr]

        return nothing
    end

    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda (1, n, n*sizeof(Float32)) kernel_shmem_dynamic_typed(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" begin
    @eval function kernel_shmem_dynamic_typevar{T}(d::CuDeviceArray{T}, n)
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

        @cuda (1, n, n*sizeof(T)) kernel_shmem_dynamic_typevar(d_a, n)
        @test reverse(a) == Array(d_a)
    end
end

@testset "alignment" begin
    # bug: used to generate align=12, which is invalid (non pow2)
    @eval function kernel_shmem_dynamic_alignment{T}(v0::T, n)
        shared = CUDAnative.@cuDynamicSharedMem(T, n)
        @inbounds shared[Cuint(1)] = v0
        return
    end

    n = 32
    T = typeof((0f0, 0f0, 0f0))
    @cuda (1, 1, n*sizeof(T)) kernel_shmem_dynamic_alignment((0f0, 0f0, 0f0), n)
end

end


@testset "static shmem" begin

@testset "statically typed" begin
    @eval function kernel_shmem_static_typed(d, n)
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

    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda (1, n) kernel_shmem_static_typed(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" begin
    @eval function kernel_shmem_static_typevar{T}(d::CuDeviceArray{T}, n)
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

        @cuda (1, n) kernel_shmem_static_typevar(d_a, n)
        @test reverse(a) == Array(d_a)
    end
end

@testset "alignment" begin
    # bug: used to generate align=12, which is invalid (non pow2)
    @eval function kernel_shmem_static_alignment{T}(v0::T)
        shared = CUDAnative.@cuStaticSharedMem(T, 32)
        @inbounds shared[Cuint(1)] = v0
        return
    end

    @cuda (1, 1) kernel_shmem_static_alignment((0f0, 0f0, 0f0))
end

end


@testset "dynamic shmem consisting of multiple arrays" begin

# common use case 1: dynamic shmem consists of multiple homogeneous arrays
#                    -> split using `view`
@testset "homogeneous" begin
    @eval function kernel_shmem_dynamic_multi_homogeneous(a, b, n)
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

    a = rand(Float32, n)
    d_a = CuArray(a)

    b = rand(Float32, n)
    d_b = CuArray(b)

    @cuda (1, n, 2*n*sizeof(Float32)) kernel_shmem_dynamic_multi_homogeneous(d_a, d_b, n)
    @test reverse(a) == Array(d_a)
    @test reverse(b) == Array(d_b)
end

# common use case 2: dynamic shmem consists of multiple heterogeneous arrays
#                    -> construct using pointer offset
@testset "heterogeneous" begin
    @eval function kernel_shmem_dynamic_multi_heterogeneous(a, b, n)
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

    a = rand(Float32, n)
    d_a = CuArray(a)

    b = rand(Int64, n)
    d_b = CuArray(b)

    @cuda (1, n, n*sizeof(Float32) + n*sizeof(Int64)) kernel_shmem_dynamic_multi_heterogeneous(d_a, d_b, n)
    @test reverse(a) == Array(d_a)
    @test reverse(b) == Array(d_b)
end

end

end



############################################################################################

@testset "data movement and conversion" begin

if capability(dev) >= v"3.0"
@testset "shuffle" begin

@eval struct AddableTuple
    x::Int32
    y::Int64
    AddableTuple(val) = new(val, val*2)
end
@eval Base.:(+)(a::AddableTuple, b::AddableTuple) = AddableTuple(a.x+b.x)

n = 14
types = [Int32, Int64, Float32, Float64, AddableTuple]

@testset "down" begin
    @eval function kernel_shuffle_down{T}(d::CuDeviceArray{T}, n)
        t = threadIdx().x
        if t <= n
            d[t] += shfl_down(d[t], unsafe_trunc(UInt32, n÷2))
        end
        return nothing
    end

    for T in types
        a = T[T(i) for i in 1:n]
        d_a = CuArray(a)

        threads = nearest_warpsize(dev, n)
        @cuda (1, threads) kernel_shuffle_down(d_a, n)

        a[1:n÷2] += a[n÷2+1:end]
        @test a == Array(d_a)
    end
end

end
end

end



############################################################################################

@testset "parallel synchronization and communication" begin

@testset "voting" begin

@testset "ballot" begin
    d_a = CuArray{UInt32}(1)

    @eval function kernel_vote_ballot(a, i)
        vote = vote_ballot(threadIdx().x == i)
        if threadIdx().x == 1
            a[1] = vote
        end

        return nothing
    end

    len = 4
    for i in 1:len
        @cuda (1,len) kernel_vote_ballot(d_a, i)
        @test Array(d_a) == [2^(i-1)]
    end
end

@testset "any" begin
    d_a = CuArray{UInt32}(1)

    @eval function kernel_vote_any(a, i)
        vote = vote_any(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end

        return nothing
    end

    @cuda (1,2) kernel_vote_any(d_a, 1)
    @test Array(d_a) == [1]

    @cuda (1,2) kernel_vote_any(d_a, 2)
    @test Array(d_a) == [1]

    @cuda (1,2) kernel_vote_any(d_a, 3)
    @test Array(d_a) == [0]
end

@testset "all" begin
    d_a = CuArray{UInt32}(1)

    @eval function kernel_vote_all(a, i)
        vote = vote_all(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end

        return nothing
    end

    @cuda (1,2) kernel_vote_all(d_a, 1)
    @test Array(d_a) == [1]

    @cuda (1,2) kernel_vote_all(d_a, 2)
    @test Array(d_a) == [0]

    @cuda (1,2) kernel_vote_all(d_a, 3)
    @test Array(d_a) == [0]
end

end

end

############################################################################################

end
