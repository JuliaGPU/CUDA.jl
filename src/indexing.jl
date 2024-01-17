# indexing


## utilities

using Base.Cartesian


## logical indexing

# we cannot use Base.LogicalIndex, which does not support indexing but requires iteration.
# TODO: it should still be possible to use the same technique;
#       Base.LogicalIndex basically contains the same as our `findall` here does.
Base.to_index(::CuArray, I::AbstractArray{Bool}) = findall(I)
if VERSION >= v"1.11.0-DEV.1157"
    Base.to_indices(A::CuArray, I::Tuple{AbstractArray{Bool}}) = (Base.to_index(A, I[1]),)
else
    Base.to_indices(A::CuArray, inds,
                    I::Tuple{Union{Array{Bool,N}, BitArray{N}}}) where {N} =
        (Base.to_index(A, I[1]),)
end


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
        threads = min(length(indices), config.threads)
        blocks = cld(length(indices), threads)
        kernel(ys, bools, indices; threads, blocks)
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
