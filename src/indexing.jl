# indexing


## utilities

using Base.Cartesian


## logical indexing

# we cannot use Base.LogicalIndex, which does not support indexing but requires iteration.
# TODO: it should still be possible to use the same technique;
#       Base.LogicalIndex basically contains the same as our `findall` here does.
Base.to_index(::AbstractGPUArray, I::AbstractArray{Bool}) = findall(I)
## same for the trailing Array{Bool} optimization (see `_maybe_linear_logical_index` in Base)
Base.to_indices(A::AbstractGPUArray, inds,
                I::Tuple{Union{Array{Bool,N}, BitArray{N}}}) where {N} =
    (Base.to_index(A, I[1]),)


## find*

function Base.findall(bools::AnyCuArray{Bool})
    I = keytype(bools)
    indices = cumsum(reshape(bools, prod(size(bools))))

    n = @allowscalar indices[end]
    ys = CuArray{I}(undef, n)

    if n > 0
        function kernel(ys::CuDeviceArray, bools, indices)
            i = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x

            @inbounds if i <= length(bools) && bools[i]
                i′ = CartesianIndices(bools)[i]
                b = indices[i]   # new position
                ys[b] = i′
            end

            return
        end

        kernel = @cuda name="findall" launch=false kernel(ys, bools, indices)
        config = launch_configuration(kernel.fun)
        threads = Base.min(length(indices), config.threads)
        blocks = cld(length(indices), threads)
        kernel(ys, bools, indices; threads=threads, blocks=blocks)
    end

    unsafe_free!(indices)

    return ys
end

function Base.findall(f::Function, A::AnyCuArray)
    bools = map(f, A)
    ys = findall(bools)
    unsafe_free!(bools)
    return ys
end

function Base.findfirst(f::Function, xs::AnyCuArray)
    indx = ndims(xs) == 1 ? (eachindex(xs), 1) :
    (CartesianIndices(xs), CartesianIndex{ndims(xs)}())
    function g(t1, t2)
        (x, i), (y, j) = t1, t2
        if i > j
            t1, t2 = t2, t1
            (x, i), (y, j) = t1, t2
        end
        x && return t1
        y && return t2
        return (false, indx[2])
    end

    res = mapreduce((x, y)->(f(x), y), g, xs, indx[1]; init = (false, indx[2]))
    res[1] === true && return res[2]
    return nothing
end

Base.findfirst(xs::AnyCuArray{Bool}) = findfirst(identity, xs)

function findminmax(minmax, binop, a::AnyCuArray; init, dims)
    function f(t1::Tuple{<:AbstractFloat,<:Any}, t2::Tuple{<:AbstractFloat,<:Any})
        (x, i), (y, j) = t1, t2
        if i > j
            t1, t2 = t2, t1
            (x, i), (y, j) = t1, t2
        end

        # Check for NaN first because NaN == NaN is false
        isnan(x) && return t1
        isnan(y) && return t2
        minmax(x, y) == x && return t1
        return t2
    end

    function f(t1, t2)
        (x, i), (y, j) = t1, t2

        binop(x, y) && return t1
        x == y && return (x, Base.min(i, j))
        return t2
    end

    indx = ndims(a) == 1 ? (eachindex(a), 1) :
                           (CartesianIndices(a), CartesianIndex{ndims(a)}())
    if dims == Colon()
        mapreduce(tuple, f, a, indx[1]; init = (init, indx[2]))
    else
        res = mapreduce(tuple, f, a, indx[1];
                        init = (init, indx[2]), dims=dims)
        vals = map(x->x[1], res)
        inds = map(x->x[2], res)
        return (vals, inds)
    end
end

Base.findmax(a::AnyCuArray; dims=:) = findminmax(max, >, a; init=typemin(eltype(a)), dims)
Base.findmin(a::AnyCuArray; dims=:) = findminmax(min, <, a; init=typemax(eltype(a)), dims)
