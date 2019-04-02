@testset "CUDA functionality" begin

############################################################################################

@testset "indexing" begin
    @on_device threadIdx().x
    @on_device blockDim().x
    @on_device blockIdx().x
    @on_device gridDim().x

    @on_device threadIdx().y
    @on_device blockDim().y
    @on_device blockIdx().y
    @on_device gridDim().y

    @on_device threadIdx().z
    @on_device blockDim().z
    @on_device blockIdx().z
    @on_device gridDim().z

    @testset "range metadata" begin
        foobar() = threadIdx().x
        ir = sprint(io->CUDAnative.code_llvm(io, foobar, Tuple{}))

        @test occursin(r"call .+ @llvm.nvvm.read.ptx.sreg.tid.x.+ !range", ir)
    end
end



############################################################################################

@testset "math" begin
    buf = CuTestArray(Float32[0])

    function kernel(a, i)
        a[1] = CUDAnative.log10(i)
        return
    end

    @cuda kernel(buf, Float32(100))
    val = Array(buf)
    @test val[1] ≈ 2.0
end



############################################################################################

@testset "formatted output" begin
    _, out = @grab_output @on_device @cuprintf("")
    @test out == ""

    endline = Sys.iswindows() ? "\r\n" : "\n"

    _, out = @grab_output @on_device @cuprintf("Testing...\n")
    @test out == "Testing...$endline"

    # narrow integer
    _, out = @grab_output @on_device @cuprintf("Testing %d...\n", Int32(42))
    @test out == "Testing 42...$endline"

    # wide integer
    _, out = @grab_output @on_device @cuprintf("Testing %ld...\n", Int64(42))
    @test out == "Testing 42...$endline"

    _, out = @grab_output if Int == Int32
        @on_device @cuprintf("Testing %lld %d...\n",
                             blockIdx().x, threadIdx().x)
    elseif Sys.iswindows()
        @on_device @cuprintf("Testing %lld %lld...\n",
                             blockIdx().x, threadIdx().x)
    else
        @on_device @cuprintf("Testing %ld %ld...\n",
                             blockIdx().x, threadIdx().x)
    end
    @test out == "Testing 1 1...$endline"

    _, out = @grab_output @on_device begin
        @cuprintf("foo")
        @cuprintf("bar\n")
    end
    @test out == "foobar$endline"

    # c argument promotions
    function kernel(A)
        @cuprintf("%f %f\n", A[1], A[1])
        return
    end
    x = CuTestArray(ones(2, 2))
    _, out = @grab_output begin
        @cuda kernel(x)
        synchronize()
    end
    @test out == "1.000000 1.000000$endline"
end



############################################################################################

@testset "assertion" begin
    function kernel(i)
        @cuassert i > 0
        @cuassert i > 0 "test"
        return
    end

    @cuda kernel(1)
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
    function kernel(d, n)
        t = threadIdx().x
        tr = n-t+1

        s = @cuDynamicSharedMem(Float32, n)
        s[t] = d[t]
        sync_threads()
        d[t] = s[tr]

        return
    end

    a = rand(Float32, n)
    d_a = CuTestArray(a)

    @cuda threads=n shmem=n*sizeof(Float32) kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" for T in [Int32, Int64, Float32, Float64]
    function kernel(d::CuDeviceArray{T}, n) where {T}
        t = threadIdx().x
        tr = n-t+1

        s = @cuDynamicSharedMem(T, n)
        s[t] = d[t]
        sync_threads()
        d[t] = s[tr]

        return
    end

    a = rand(T, n)
    d_a = CuTestArray(a)

    @cuda threads=n shmem=n*sizeof(T) kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "alignment" begin
    # bug: used to generate align=12, which is invalid (non pow2)
    function kernel(v0::T, n) where {T}
        shared = @cuDynamicSharedMem(T, n)
        @inbounds shared[Cuint(1)] = v0
        return
    end

    n = 32
    typ = typeof((0f0, 0f0, 0f0))
    @cuda shmem=n*sizeof(typ) kernel((0f0, 0f0, 0f0), n)
end

end


@testset "static shmem" begin

@testset "statically typed" begin
    function kernel(d, n)
        t = threadIdx().x
        tr = n-t+1

        s = @cuStaticSharedMem(Float32, 1024)
        s2 = @cuStaticSharedMem(Float32, 1024)  # catch aliasing

        s[t] = d[t]
        s2[t] = 2*d[t]
        sync_threads()
        d[t] = s[tr]

        return
    end

    a = rand(Float32, n)
    d_a = CuTestArray(a)

    @cuda threads=n kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" for typ in [Int32, Int64, Float32, Float64]
    function kernel(d::CuDeviceArray{T}, n) where {T}
        t = threadIdx().x
        tr = n-t+1

        s = @cuStaticSharedMem(T, 1024)
        s2 = @cuStaticSharedMem(T, 1024)  # catch aliasing

        s[t] = d[t]
        s2[t] = d[t]
        sync_threads()
        d[t] = s[tr]

        return
    end

    a = rand(typ, n)
    d_a = CuTestArray(a)

    @cuda threads=n kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "alignment" begin
    # bug: used to generate align=12, which is invalid (non pow2)
    function kernel(v0::T) where {T}
        shared = CUDAnative.@cuStaticSharedMem(T, 32)
        @inbounds shared[Cuint(1)] = v0
        return
    end

    @cuda kernel((0f0, 0f0, 0f0))
end

end


@testset "dynamic shmem consisting of multiple arrays" begin

# common use case 1: dynamic shmem consists of multiple homogeneous arrays
#                    -> split using `view`
@testset "homogeneous" begin
    function kernel(a, b, n)
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

        return
    end

    a = rand(Float32, n)
    d_a = CuTestArray(a)

    b = rand(Float32, n)
    d_b = CuTestArray(b)

    @cuda threads=n shmem=2*n*sizeof(Float32) kernel(d_a, d_b, n)
    @test reverse(a) == Array(d_a)
    @test reverse(b) == Array(d_b)
end

# common use case 2: dynamic shmem consists of multiple heterogeneous arrays
#                    -> construct using pointer offset
@testset "heterogeneous" begin
    function kernel(a, b, n)
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

        return
    end

    a = rand(Float32, n)
    d_a = CuTestArray(a)

    b = rand(Int64, n)
    d_b = CuTestArray(b)

    @cuda threads=n shmem=(n*sizeof(Float32) + n*sizeof(Int64)) kernel(d_a, d_b, n)
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
Base.:(+)(a::AddableTuple, b::AddableTuple) = AddableTuple(a.x+b.x)

n = 14

@testset "down" for T in [Int32, Int64, Float32, Float64, AddableTuple]
    function kernel(d::CuDeviceArray{T}, n) where {T}
        t = threadIdx().x
        if t <= n
            d[t] += shfl_down(d[t], n÷2)
        end
        return
    end

    a = T[T(i) for i in 1:n]
    d_a = CuTestArray(a)

    threads = nearest_warpsize(dev, n)
    @cuda threads=threads kernel(d_a, n)

    a[1:n÷2] += a[n÷2+1:end]
    @test a == Array(d_a)
end

end
end

end



############################################################################################

@testset "clock and nanosleep" begin
@on_device clock(UInt32)
@on_device clock(UInt64)
@on_device nanosleep(UInt32(16))
end

@testset "parallel synchronization and communication" begin

@on_device sync_threads()
@on_device sync_threads_count(Int32(1))
@on_device sync_threads_count(true)
@on_device sync_threads_and(Int32(1))
@on_device sync_threads_and(true)
@on_device sync_threads_or(Int32(1))
@on_device sync_threads_or(true)

@testset "llvm ir barrier int" begin
    function kernel_barrier_count(an_array_of_1)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if sync_threads_count(an_array_of_1[i]) == 3
            an_array_of_1[i] += 1
        end
        return nothing
    end

    b_in = Int32[1, 1, 1, 0, 0]  # 3 true
    d_b = CuTestArray(b_in)
    @cuda threads=5 kernel_barrier_count(d_b)
    b_out = Array(d_b)

    @test b_out == b_in .+ 1

    function kernel_barrier_and(an_array_of_1)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if sync_threads_and(an_array_of_1[i]) > 0
            an_array_of_1[i] += 1
        end
        return nothing
    end

    a_in = Int32[1, 1, 1, 1, 1]  # all true
    d_a = CuTestArray(a_in)
    @cuda threads=5 kernel_barrier_and(d_a)
    a_out = Array(d_a)

    @test a_out == a_in .+ 1

    function kernel_barrier_or(an_array_of_1)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if sync_threads_or(an_array_of_1[i]) > 0
            an_array_of_1[i] += 1
        end
        return nothing
    end

    c_in = Int32[1, 0, 0, 0, 0]  # 1 true
    d_c = CuTestArray(c_in)
    @cuda threads=5 kernel_barrier_or(d_c)
    c_out = Array(d_c)

    @test c_out == c_in .+ 1
end

@testset "llvm ir barrier bool" begin
    function kernel_barrier_count(an_array_of_1)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if sync_threads_count(an_array_of_1[i] > 0) == 3
            an_array_of_1[i] += 1
        end
        return nothing
    end

    b_in = Int32[1, 1, 1, 0, 0]  # 3 true
    d_b = CuTestArray(b_in)
    @cuda threads=5 kernel_barrier_count(d_b)
    b_out = Array(d_b)

    @test b_out == b_in .+ 1

    function kernel_barrier_and(an_array_of_1)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if sync_threads_and(an_array_of_1[i] > 0)
            an_array_of_1[i] += 1
        end
        return nothing
    end

    a_in = Int32[1, 1, 1, 1, 1]  # all true
    d_a = CuTestArray(a_in)
    @cuda threads=5 kernel_barrier_and(d_a)
    a_out = Array(d_a)

    @test a_out == a_in .+ 1

    function kernel_barrier_or(an_array_of_1)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if sync_threads_or(an_array_of_1[i] > 0)
            an_array_of_1[i] += 1
        end
        return nothing
    end

    c_in = Int32[1, 0, 0, 0, 0]  # 1 true
    d_c = CuTestArray(c_in)
    @cuda threads=5 kernel_barrier_or(d_c)
    c_out = Array(d_c)

    @test c_out == c_in .+ 1
end

@testset "grid groups sync" begin
    function kernel_vadd(a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        grid_handle = this_grid()
        c[i] = a[i] + b[i]
        sync_grid(grid_handle)
        c[i] = c[1]
        return nothing
    end

    a = round.(rand(Float32, (300, 40)) * 100)
    b = round.(rand(Float32, (300, 40)) * 100)
    c = zeros(Float32, (300, 40))
    d_a = CuTestArray(a)
    d_b = CuTestArray(b)
    d_c = CuTestArray(c)  # output array
    @cuda cg=true threads=600 blocks=20 kernel_vadd(d_a, d_b, d_c)
    c = Array(d_c)
    @test all(c[1] .== c)
end

@on_device sync_warp()
@on_device sync_warp(0xffffffff)
@on_device threadfence_block()
@on_device threadfence()
@on_device threadfence_system()

@testset "voting" begin

@testset "ballot" begin
    d_a = CuTestArray(UInt32[0])

    function kernel(a, i)
        vote = vote_ballot(threadIdx().x == i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    len = 4
    for i in 1:len
        @cuda threads=len kernel(d_a, i)
        @test Array(d_a) == [2^(i-1)]
    end
end

@testset "any" begin
    d_a = CuTestArray(UInt32[0])

    function kernel(a, i)
        vote = vote_any(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel(d_a, 1)
    @test Array(d_a) == [1]

    @cuda threads=2 kernel(d_a, 2)
    @test Array(d_a) == [1]

    @cuda threads=2 kernel(d_a, 3)
    @test Array(d_a) == [0]
end

@testset "all" begin
    d_a = CuTestArray(UInt32[0])

    function kernel(a, i)
        vote = vote_all(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel(d_a, 1)
    @test Array(d_a) == [1]

    @cuda threads=2 kernel(d_a, 2)
    @test Array(d_a) == [0]

    @cuda threads=2 kernel(d_a, 3)
    @test Array(d_a) == [0]
end

end

end

############################################################################################

@testset "libcudadevrt" begin
    kernel() = (CUDAnative.synchronize(); nothing)
    @cuda kernel()
end

############################################################################################

end
