# hostcall: calling host functions from kernels

group = addgroup!(SUITE, "hostcall")

hostcall_identity(x::Int) = x

# launch overhead of a kernel that is armed for hostcalls (but never calls)
function hostcall_armed_kernel(out, flag)
    if flag
        out[1] = @hostcall hostcall_identity(1)::Int
    end
    return
end
out = CUDA.zeros(Int, 1)
group["launch_armed"] = @benchmarkable @cuda hostcall_armed_kernel($out, false)

# a single blocking call per warp; latency dominated
function hostcall_blocking_kernel(out)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    out[i] = @hostcall hostcall_identity(Int(i))::Int
    return
end
out1 = CUDA.zeros(Int, 32)
group["blocking_1warp"] = @async_benchmarkable @cuda threads=32 hostcall_blocking_kernel($out1)
out512 = CUDA.zeros(Int, 32 * 512)
group["blocking_512warps"] = @async_benchmarkable @cuda threads=256 blocks=64 hostcall_blocking_kernel($out512)

# fire-and-forget calls, drained by synchronize()
function hostcall_async_kernel(out)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @hostcall async=true hostcall_identity(Int(i))
    return
end
group["async_1warp"] = @async_benchmarkable @cuda threads=32 hostcall_async_kernel($out1)
group["async_512warps"] = @async_benchmarkable @cuda threads=256 blocks=64 hostcall_async_kernel($out512)
