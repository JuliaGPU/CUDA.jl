@testset "cooperative groups" begin

###########################################################################################

@testset "thread blocks" begin
    # simple reverse kernel, but using cooperative groups instead of indexing intrinsics
    T = Int32
    n = 256

    function kernel(d::CuDeviceArray{T}, n) where {T}
        block = CG.this_thread_block()

        t = CG.thread_rank(block)
        tr = n-t+1

        s = @inbounds CuDynamicSharedArray(T, n)
        @inbounds s[t] = d[t]
        CG.sync(block)
        @inbounds d[t] = s[tr]

        return
    end

    a = rand(T, n)
    d_a = CuArray(a)

    @cuda threads=n shmem=n*sizeof(T) kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

###########################################################################################

if capability(device()) >= v"6.0" && attribute(device(), CUDA.DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) == 1
@testset "grid groups" begin
    function kernel_vadd(a, b, c)
        grid = CG.this_grid()
        i = CG.thread_rank(grid)
        c[i] = a[i] + b[i]
        CG.sync(grid)
        c[i] = c[1]
        return nothing
    end

    # cooperative kernels are limited in the number of blocks that can be launched
    # (the occupancy API could be used to calculate how many blocks can fit per SM,
    #  but that doesn't matter for the tests, so we assume a single block per SM.)
    maxBlocks = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    kernel = cufunction(kernel_vadd, NTuple{3, CuDeviceArray{Float32,2,AS.Global,Int}})
    maxThreads = CUDA.maxthreads(kernel)

    a = rand(Float32, maxBlocks, maxThreads)
    b = rand(Float32, size(a)) * 100
    c = similar(a)
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)  # output array

    @cuda cooperative=true threads=maxThreads blocks=maxBlocks kernel_vadd(d_a, d_b, d_c)

    c = Array(d_c)
    @test all(c[1] .== c)
end
end

###########################################################################################

@testset "coalesced groups" begin

@testset "shuffle" begin
    function reverse_kernel(d, lower, upper)
        cta = CG.this_thread_block()
        I = CG.thread_rank(cta)
        if lower <= threadIdx().x <= upper
            warp = CG.coalesced_threads()
            i = CG.thread_rank(warp)
            j = CG.num_threads(warp) - i + 1

            d[I] = CG.shfl(warp, d[I], j)
        end

        return
    end

    function shift_up_kernel(d, lower, upper, delta)
        cta = CG.this_thread_block()
        I = CG.thread_rank(cta)
        if lower <= threadIdx().x <= upper
            warp = CG.coalesced_threads()
            d[I] = CG.shfl_up(warp, d[I], delta)
        end
        return
    end

    function shift_down_kernel(d, lower, upper, delta)
        cta = CG.this_thread_block()
        I = CG.thread_rank(cta)
        if lower <= threadIdx().x <= upper
            warp = CG.coalesced_threads()
            d[I] = CG.shfl_down(warp, d[I], delta)
        end
        return
    end

    warpsize = CUDA.warpsize(device())
    delta = rand(1:5)
    lower, upper = 20, 30

    @testset for T in [UInt8, UInt16, UInt32, UInt64, UInt128,
                       Int8, Int16, Int32, Int64, Int128,
                       Float16, Float32, Float64,
                       ComplexF32, ComplexF64, Bool]
        a = rand(T, warpsize)

        # cooperative shuffles are implemented differently when they operate
        # on the entire warp, or only a subset, so we test both cases.

        # reverse the entire array
        d_a = CuArray(a)
        @cuda threads=warpsize reverse_kernel(d_a, 1, warpsize)
        @test Array(d_a) == reverse(a)

        # shift up the entire array
        d_a = CuArray(a)
        @cuda threads=warpsize shift_up_kernel(d_a, 1, warpsize, delta)
        expected = [a[1:delta]; a[1:end-delta]]
        @test Array(d_a) == expected

        # shift down the entire array
        d_a = CuArray(a)
        @cuda threads=warpsize shift_down_kernel(d_a, 1, warpsize, delta)
        expected = [a[delta+1:end]; a[end-delta+1:end]]
        @test Array(d_a) == expected

        # reverse only a part
        d_a = CuArray(a)
        @cuda threads=warpsize reverse_kernel(d_a, lower, upper)
        @test Array(d_a)[1:lower-1] == a[1:lower-1]
        @test Array(d_a)[lower:upper] == reverse(a[lower:upper])
        @test Array(d_a)[upper+1:end] == a[upper+1:end]

        # shift up only a part
        d_a = CuArray(a)
        @cuda threads=warpsize shift_up_kernel(d_a, lower, upper, delta)
        expected = [a[1:lower+delta-1]; a[lower:upper-delta]; a[upper+1:end]]
        @test Array(d_a) == expected

        # shift down only a part
        d_a = CuArray(a)
        @cuda threads=warpsize shift_down_kernel(d_a, lower, upper, delta)
        expected = [a[1:lower-1]; a[lower+delta:upper]; a[upper-delta+1:end]]
        @test Array(d_a) == expected
    end
end

@testset "warp-aggregated atomic increment" begin
    # from https://developer.nvidia.com/blog/cooperative-groups/
    @inline function atomic_agg_inc!(ptr)
        g = CG.coalesced_threads()
        prev = Int32(0)

        # elect the first active thread to perform atomic add
        if CG.thread_rank(g) == 1
            prev = CUDA.atomic_add!(ptr, Int32(CG.num_threads(g)))
        end

        # broadcast previous value within the warp
        # and add each active thread’s rank to it
        CG.thread_rank(g) - 1i32 + CG.shfl(g, prev, 1i32)
    end
    function kernel(arr)
        if threadIdx().x % 2 == 0
            atomic_agg_inc!(pointer(arr))
        end
        return
    end

    x = CuArray(Int32[0])
    @cuda threads=10 kernel(x)
    @test Array(x)[] == 5
end
end

###########################################################################################

@testset "data transfer" begin

@testset "memcpy_async" begin
    function kernel(input::AbstractArray{T}, output::AbstractArray{T},
                    elements_per_copy, group_ctor) where {T}
        # simple kernel that copies global memory, staging through a shared memory buffer,
        # using memcpy_async to perform the copies. the kernel only uses a single block,
        # as the memcpy_async computations only work on (tiled) thread blocks.
        tb = group_ctor()

        local_smem = CuDynamicSharedArray(T, elements_per_copy)
        bytes_per_copy = sizeof(local_smem)

        i = 1
        while i <= length(input)
            # memcpy_async is a collective operation, so we call it identically on all
            # threads and the implementation takes care of the rest

            checkbounds(input, i:i+elements_per_copy-1)
            checkbounds(output, i:i+elements_per_copy-1)
            checkbounds(local_smem, 1:elements_per_copy)

            # this copy can sometimes be accelerated
            CG.memcpy_async(tb, pointer(local_smem), pointer(input, i), bytes_per_copy)
            CG.wait(tb)

            # this copy is always a simple element-wise operation
            CG.memcpy_async(tb, pointer(output, i), pointer(local_smem), bytes_per_copy)
            CG.wait(tb)

            i += elements_per_copy
        end
    end

    @testset for T in [UInt8, UInt16, UInt32, UInt64, UInt128,
                       Int8, Int16, Int32, Int64, Int128,
                       Float16, Float32, Float64,
                       ComplexF32, ComplexF64, Bool],
                 threads in [1, 16, 32, 128],
                 elements_per_copy in [128, 256, 512],
                 group_ctor in [CG.coalesced_threads, CG.this_thread_block]
        data = rand(T, 4096)
        input = CuArray(data)
        output = similar(input)
        shmem = elements_per_copy * sizeof(T)
        @cuda threads shmem kernel(input, output, elements_per_copy, group_ctor)
        @test Array(output) == data skip=sanitize
        # XXX: this occasionally fails under compute-sanitizer
    end
end

end

###########################################################################################

end
