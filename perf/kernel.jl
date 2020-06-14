group = addgroup!(SUITE, "kernel")

dummy_kernel() = nothing
group["launch"] = @benchmarkable @cuda dummy_kernel()

wanted_threads = 10000
function configurator(kernel)
    config = launch_configuration(kernel.fun)

    threads = Base.min(wanted_threads, config.threads)
    blocks = cld(wanted_threads, threads)

    return (threads=threads, blocks=blocks)
end
group["occupancy"] = @benchmarkable @cuda config=$configurator dummy_kernel()

src = CUDA.rand(Float32, 512, 1000)
dest = similar(src)
function indexing_kernel(dest, src)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds dest[i] = src[i]
    return
end
group["indexing"] = @benchmarkable CUDA.@sync @cuda threads=size(src,1) blocks=size(src,2) $indexing_kernel($dest, $src)

function checked_indexing_kernel(dest, src)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    dest[i] = src[i]
    return
end
group["indexing_checked"] = @benchmarkable CUDA.@sync @cuda threads=size(src,1) blocks=size(src,2) $checked_indexing_kernel($dest, $src)
