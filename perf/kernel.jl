using CUDA: i32

group = addgroup!(SUITE, "kernel")

dummy_kernel() = nothing
group["launch"] = @benchmarkable @cuda dummy_kernel()

wanted_threads = 10000
group["occupancy"] = @benchmarkable begin
    kernel = @cuda launch=false dummy_kernel()
    config = launch_configuration(kernel.fun)
    threads = Base.min($wanted_threads, config.threads)
    blocks = cld($wanted_threads, threads)
end

src = CUDA.rand(Float32, 512, 1000)
dest = similar(src)
function indexing_kernel(dest, src)
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    @inbounds dest[i] = src[i]
    return
end
group["indexing"] = @async_benchmarkable @cuda threads=size(src,1) blocks=size(src,2) $indexing_kernel($dest, $src)

function checked_indexing_kernel(dest, src)
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    dest[i] = src[i]
    return
end
group["indexing_checked"] = @async_benchmarkable @cuda threads=size(src,1) blocks=size(src,2) $checked_indexing_kernel($dest, $src)

function rand_kernel(dest::AbstractArray{T}) where {T}
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    dest[i] = rand(T)
    return
end
group["rand"] = @async_benchmarkable @cuda threads=size(src,1) blocks=size(src,2) $rand_kernel($dest)
