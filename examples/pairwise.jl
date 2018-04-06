# calculate pairwise distance between every point in a vector

using CUDAnative, CUDAdrv


function haversine_cpu(lat1::Float32, lon1::Float32, lat2::Float32, lon2::Float32, radius::Float32)
    c1 = cospi(lat1 / 180.0f0)
    c2 = cospi(lat2 / 180.0f0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d1 = sinpi(dlat / 360.0f0)
    d2 = sinpi(dlon / 360.0f0)
    t = d2 * d2 * c1 * c2
    a = d1 * d1 + t
    c = 2.0f0 * asin(min(1.0f0, sqrt(a)))
    return radius * c
end

function pairwise_dist_cpu(lat::Vector{Float32}, lon::Vector{Float32})
    # allocate
    n = length(lat)
    rowresult = Array{Float32}(undef, n, n)
    
    # brute force fill in each cell
    for i in 1:n, j in 1:n
        @inbounds rowresult[i, j] = haversine_cpu(lat[i], lon[i], lat[j], lon[j] , 6372.8f0)
    end
    
    return rowresult    
end

# from https://devblogs.nvidia.com/parallelforall/fast-great-circle-distance-calculation-cuda-c/
function haversine_gpu(lat1::Float32, lon1::Float32, lat2::Float32, lon2::Float32, radius::Float32)
    # XXX: need to prefix math intrinsics with CUDAnative
    c1 = CUDAnative.cospi(lat1 / 180.0f0)
    c2 = CUDAnative.cospi(lat2 / 180.0f0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d1 = CUDAnative.sinpi(dlat / 360.0f0)
    d2 = CUDAnative.sinpi(dlon / 360.0f0)
    t = d2 * d2 * c1 * c2
    a = d1 * d1 + t
    c = 2.0f0 * CUDAnative.asin(CUDAnative.min(1.0f0, CUDAnative.sqrt(a)))
    return radius * c
end

# pairwise distance calculation kernel
function pairwise_dist_kernel(lat::CuDeviceVector{Float32}, lon::CuDeviceVector{Float32},
                              rowresult::CuDeviceMatrix{Float32}, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        # store to shared memory
        shmem = @cuDynamicSharedMem(Float32, 2*blockDim().x + 2*blockDim().y)
        if threadIdx().y == 1
            shmem[threadIdx().x] = lat[i]
            shmem[blockDim().x + threadIdx().x] = lon[i]
        end
        if threadIdx().x == 1
            shmem[2*blockDim().x + threadIdx().y] = lat[j]
            shmem[2*blockDim().x + blockDim().y + threadIdx().y] = lon[j]
        end
        sync_threads()

        # load from shared memory
        lat_i = shmem[threadIdx().x]
        lon_i = shmem[blockDim().x + threadIdx().x]
        lat_j = shmem[2*blockDim().x + threadIdx().y]
        lon_j = shmem[2*blockDim().x + blockDim().y + threadIdx().y]

        @inbounds rowresult[i, j] = haversine_gpu(lat_i, lon_i, lat_j, lon_j, 6372.8f0)
    end
end

function pairwise_dist_gpu(lat::Vector{Float32}, lon::Vector{Float32})
    # upload
    lat_gpu = CuArray(lat)
    lon_gpu = CuArray(lon)

    # allocate
    n = length(lat)
    rowresult_gpu = CuArray{Float32}(n, n)

    # calculate launch configuration
    # NOTE: we want our launch configuration to be as square as possible,
    #       because that minimizes shared memory usage
    ctx = CuCurrentContext()
    dev = device(ctx)
    total_threads = min(n, attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))
    threads_x = floor(Int, sqrt(total_threads))
    threads_y = total_threads ÷ threads_x
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, n ./ threads)

    # calculate size of dynamic shared memory
    shmem = 2 * sum(threads) * sizeof(Float32)

    @cuda blocks=blocks threads=threads shmem=shmem pairwise_dist_kernel(lat_gpu, lon_gpu, rowresult_gpu, n)

    return Array(rowresult_gpu)
end


# generate reasonable data
const n = 10000
const lat = rand(Float32, n) .* 45
const lon = rand(Float32, n) .* -120

using Test

@test pairwise_dist_cpu(lat, lon) ≈ pairwise_dist_gpu(lat, lon)
