# indexing


## utilities

using Base.Cartesian


## logical indexing

Base.getindex(xs::CuArray, bools::AbstractArray{Bool}) = getindex(xs, CuArray(bools))

function Base.getindex(xs::CuArray{T}, bools::CuArray{Bool}) where {T}
  bools = reshape(bools, prod(size(bools)))
  indices = cumsum(bools)  # unique indices for elements that are true

  n = @allowscalar indices[end]  # number that are true
  ys = CuArray{T}(undef, n)

  if n > 0
    function kernel(ys::CuDeviceArray{T}, xs::CuDeviceArray{T}, bools, indices)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if i <= length(xs) && bools[i]
            b = indices[i]   # new position
            ys[b] = xs[i]
        end

        return
    end

    function configurator(kernel)
        config = launch_configuration(kernel.fun)

        threads = Base.min(length(indices), config.threads)
        blocks = cld(length(indices), threads)

        return (threads=threads, blocks=blocks)
    end

    @cuda name="logical_getindex" config=configurator kernel(ys, xs, bools, indices)
  end

  unsafe_free!(indices)

  return ys
end


## find*

function Base.findall(bools::CuArray{Bool})
    I = keytype(bools)
    indices = cumsum(reshape(bools, prod(size(bools))))

    n = @allowscalar indices[end]
    ys = CuArray{I}(undef, n)

    if n > 0
        function kernel(ys::CuDeviceArray, bools, indices)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if i <= length(bools) && bools[i]
                i′ = CartesianIndices(bools)[i]
                b = indices[i]   # new position
                ys[b] = i′
            end

            return
        end

        function configurator(kernel)
            config = launch_configuration(kernel.fun)

            threads = Base.min(length(indices), config.threads)
            blocks = cld(length(indices), threads)

            return (threads=threads, blocks=blocks)
        end

        @cuda name="findall" config=configurator kernel(ys, bools, indices)
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

function Base.findfirst(testf::Function, xs::CuArray)
    I = keytype(xs)

    y = CuArray([typemax(Int)])

    function kernel(y::CuDeviceArray, xs::CuDeviceArray)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if i <= length(xs) && testf(xs[i])
            @atomic y[1] = Base.min(y[1], i)
        end

        return
    end

    function configurator(kernel)
        config = launch_configuration(kernel.fun)

        threads = Base.min(length(xs), config.threads)
        blocks = cld(length(xs), threads)

        return (threads=threads, blocks=blocks)
    end

    @cuda name="findfirst" config=configurator kernel(y, xs)

    first_i = @allowscalar y[1]
    return first_i == typemax(Int) ? nothing : keys(xs)[first_i]
end

Base.findfirst(xs::CuArray{Bool}) = findfirst(identity, xs)

function Base.findmin(a::CuArray; dims=:)
    if dims == Colon()
        m = minimum(a)
        i = findfirst(x->x==m, a)
        return m,i
    else
        minima = minimum(a; dims=dims)
        i = findfirstval(minima, a)
        return minima,i
    end
end

function Base.findmax(a::CuArray; dims=:)
    if dims == Colon()
        m = maximum(a)
        i = findfirst(x->x==m, a)
        return m,i
    else
        maxima = maximum(a; dims=dims)
        i = findfirstval(maxima, a)
        return maxima,i
    end
end

function findfirstval(vals::CuArray, xs::CuArray)
    ## find the first matching element

    # NOTE: this kernel performs global atomic operations for the sake of simplicity.
    #       if this turns out to be a bottleneck, we will need to cache in local memory.
    #       that requires the dimension-under-reduction to be iterated in first order.
    #       this can be done by splitting the iteration domain eagerly; see the
    #       accumulate kernel for an example, or git history from before this comment.

    indices = fill(typemax(Int), size(vals))

    function kernel(xs, vals, indices)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x

        R = CartesianIndices(xs)

        if i <= length(R)
            I = R[i]
            Jmax = last(CartesianIndices(vals))
            J = Base.min(I, Jmax)

            @inbounds if xs[I] == vals[J]
                I′ = LinearIndices(xs)[I]      # atomic_min only works with integers
                J′ = LinearIndices(indices)[J] # FIXME: @atomic doesn't handle array ref with CartesianIndices
                @atomic indices[J′] = Base.min(indices[J′], I′)
            end
        end

        return
    end

    function configurator(kernel)
        config = launch_configuration(kernel.fun)

        threads = Base.min(length(xs), config.threads)
        blocks = cld(length(xs), threads)

        return (threads=threads, blocks=blocks)
    end

    @cuda config=configurator kernel(xs, vals, indices)


    ## convert the linear indices to an appropriate type

    kt = keytype(xs)

    if kt == Int
        return indices
    else
        indices′ = CuArray{kt}(undef, size(indices))
        broadcast!(indices′, indices, Ref(keys(xs))) do index, keys
            keys[index]
        end

        return indices′
    end
end
