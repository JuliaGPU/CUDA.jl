import GPUArrays: allowscalar, @allowscalar

function _getindex(xs::CuArray{T}, i::Integer) where T
  buf = Array{T}(undef)
  copyto!(buf, 1, xs, i, 1)
  buf[]
end

function _setindex!(xs::CuArray{T}, v::T, i::Integer) where T
  copyto!(xs, i, T[v], 1, 1)
end


## logical indexing

Base.getindex(xs::CuArray, bools::AbstractArray{Bool}) = getindex(xs, CuArray(bools))

function Base.getindex(xs::CuArray{T}, bools::CuArray{Bool}) where {T}
  bools = reshape(bools, prod(size(bools)))
  indices = cumsum(bools)  # unique indices for elements that are true

  n = _getindex(indices, length(indices))  # number that are true
  ys = CuArray{T}(undef, n)

  if n > 0
    function kernel(ys::CuDeviceArray{T}, xs::CuDeviceArray{T}, bools, indices)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if i <= length(xs) && bools[i]
            b = indices[i]   # new position
            ys[b] = xs[i]

        end

        return
    end

    function configurator(kernel)
        fun = kernel.fun
        config = launch_configuration(fun)
        blocks = cld(length(indices), config.threads)

        return (threads=config.threads, blocks=blocks)
    end

    @cuda config=configurator kernel(ys, xs, bools, indices)
  end

  unsafe_free!(indices)

  return ys
end


## findall

function Base.findall(bools::CuArray{Bool})
    indices = cumsum(bools)

    n = _getindex(indices, length(indices))
    ys = CuArray{Int}(undef, n)

    if n > 0
        num_threads = min(n, 256)
        num_blocks = ceil(Int, length(indices) / num_threads)

        function kernel(ys::CuDeviceArray{Int}, bools, indices)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            if i <= length(bools) && bools[i]
                b = indices[i]   # new position
                ys[b] = i

            end

            return
        end

        function configurator(kernel)
            fun = kernel.fun
            config = launch_configuration(fun)
            blocks = cld(length(indices), config.threads)

            return (threads=config.threads, blocks=blocks)
        end

        @cuda config=configurator kernel(ys, bools, indices)
    end

    unsafe_free!(indices)

    return ys
end

function Base.findall(f::Function, A::CuArray)
    bools = map(f, A)
    ys = findall(bools)
    unsafe_free!(bools)
    return ys
end
