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
    @test_approx_eq val[1] 2.0

    free(buf)
end


## shared memory

let
    @target ptx function kernel_reverse{T}(d::CuDeviceArray{T}, n)
        t = threadIdx().x
        tr = n-t+1

        s = cuSharedMem(T)
        s[t] = d[t]
        sync_threads()
        d[t] = s[tr]

        return nothing
    end

    n = 1024
    types = [Int64, Float32, Float64]

    for T in types
        a = rand(T, n)
        d_a = CuArray(a)

        @cuda (1, n, n*sizeof(T)) kernel_reverse(d_a, n)

        @assert reverse(a) == Array(d_a)
    end
end


destroy(ctx)
