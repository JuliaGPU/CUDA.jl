# indexing

using GPUArrays: allowscalar, @allowscalar


## utilities

using Base.Cartesian

function cudims(n::Integer)
  threads = min(n, 256)
  ceil(Int, n / threads), threads
end

cudims(a::AbstractArray) = cudims(length(a))

# COV_EXCL_START
@inline ind2sub_(a::AbstractArray{T,0}, i) where T = ()
@inline ind2sub_(a, i) = Tuple(CartesianIndices(a)[i])

macro cuindex(A)
  quote
    A = $(esc(A))
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(A) && return
    ind2sub_(A, i)
  end
end

@generated function nindex(i::T, ls::NTuple{N,T}) where {N,T}
  na = one(i)
  quote
    Base.@_inline_meta
    $(foldr((n, els) -> :(i ≤ ls[$n] ? ($n, i) : (i -= ls[$n]; $els)), :($na, $na), one(i):i(N)))
  end
end

@inline function catindex(dim, I::NTuple{N}, shapes) where N
  @inbounds x, i = nindex(I[dim], getindex.(shapes, dim))
  x, ntuple(n -> n == dim ? i : I[n], Val{N})
end
# COV_EXCL_STOP

function growdims(dim, x)
  if ndims(x) >= dim
    x
  else
    reshape(x, size.((x,), 1:dim)...)
  end
end

function _cat(dim, dest, xs...)
  function kernel(dim, dest, xs)
    I = @cuindex dest
    @inbounds n, I′ = catindex(dim, Int.(I), size.(xs))
    @inbounds dest[I...] = xs[n][I′...]
    return
  end
  xs = growdims.(dim, xs)
  blk, thr = cudims(dest)
  @cuda blocks=blk threads=thr kernel(dim, dest, xs)
  return dest
end

function Base.cat_t(dims::Integer, T::Type, x::CuArray, xs::CuArray...)
  catdims = Base.dims2cat(dims)
  shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
  dest = Base.cat_similar(x, T, shape)
  _cat(dims, dest, x, xs...)
end

Base.vcat(xs::CuArray...) = cat(xs..., dims=1)
Base.hcat(xs::CuArray...) = cat(xs..., dims=2)


## non-asserting indexing

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

        @inbounds if i <= length(xs) && bools[i]
            b = indices[i]   # new position
            ys[b] = xs[i]
        end

        return
    end

    function configurator(kernel)
        config = launch_configuration(kernel.fun)

        threads = min(length(indices), config.threads)
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

    n = _getindex(indices, length(indices))
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

            threads = min(length(indices), config.threads)
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
            CUDAnative.@atomic y[1] = min(y[1], i)
        end

        return
    end

    function configurator(kernel)
        config = launch_configuration(kernel.fun)

        threads = min(length(xs), config.threads)
        blocks = cld(length(xs), threads)

        return (threads=threads, blocks=blocks)
    end

    @cuda name="findfirst" config=configurator kernel(y, xs)

    first_i = _getindex(y, 1)
    return keys(xs)[first_i]
end

Base.findfirst(xs::CuArray{Bool}) = findfirst(identity, xs)

function Base.findfirst(vals::CuArray, xs::CuArray)
    # figure out which dimension was reduced
    @assert ndims(vals) == ndims(xs)
    dims = [i for i in 1:ndims(xs) if size(xs,i)!=1 && size(vals,i)==1]
    @assert length(dims) == 1
    dim = dims[1]


    ## find the first matching element

    indices = fill(typemax(Int), size(vals))

    # iteration domain across the main dimension
    Rdim = CartesianIndices((size(xs, dim),))

    # iteration domain for the other dimensions
    Rpre = CartesianIndices(size(xs)[1:dim-1])
    Rpost = CartesianIndices(size(xs)[dim+1:end])
    Rother = CartesianIndices((length(Rpre), length(Rpost)))

    function kernel(xs, vals, indices, Rdim, Rpre, Rpost, Rother)
        # iterate the main dimension using threads and the first block dimension
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        # iterate the other dimensions using the remaining block dimensions
        j = (blockIdx().z-1) * gridDim().y + blockIdx().y

        if i <= length(Rdim) && j <= length(Rother)
            I = Rother[j]
            Ipre = Rpre[I[1]]
            Ipost = Rpost[I[2]]

            @inbounds if xs[Ipre, i, Ipost] == vals[Ipre, Ipost]
                full_index = LinearIndices(xs)[Ipre, i, Ipost]   # atomic_min only works with integers
                reduced_index = LinearIndices(indices)[Ipre, Ipost] # FIXME: @atomic doesn't handle array ref with CartesianIndices
                CUDAnative.@atomic indices[reduced_index] = min(indices[reduced_index], full_index)
            end
        end

        return
    end

    function configurator(kernel)
        # what's a good launch configuration for this kernel?
        config = launch_configuration(kernel.fun)

        # blocks to cover the main dimension
        threads = min(length(Rdim), config.threads)
        blocks_dim = cld(length(Rdim), threads)
        # NOTE: the grid X dimension is virtually unconstrained

        # blocks to cover the remaining dimensions
        dev = CUDAdrv.device(kernel.fun.mod.ctx)
        max_other_blocks = attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
        blocks_other = (min(length(Rother), max_other_blocks),
                        cld(length(Rother), max_other_blocks))

        return (threads=threads, blocks=(blocks_dim, blocks_other...))
    end

    @cuda config=configurator kernel(xs, vals, indices, Rdim, Rpre, Rpost, Rother)


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

function Base.findmin(a::CuArray; dims=:)
    if dims == Colon()
        m = minimum(a)
        i = findfirst(x->x==m, a)
        return m,i
    else
        minima = minimum(a; dims=dims)
        i = findfirst(minima, a)
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
        i = findfirst(maxima, a)
        return maxima,i
    end
end
