# Bounds checked on the converted arguments before every launch. GPUCompiler supplies
# the common indexing-object policies, emits the assumptions, and narrows index arithmetic.
# Keep byte sizes unbounded: an array with 32-bit indices can occupy more than 2 GiB.
function GPUCompiler.specialization_ranges(::Type{<:CuDeviceArray{T,N}}) where {T,N}
    ranges = ()
    for i in 1:N
        ranges = (ranges..., (:dims, i) => 0:typemax(Int32))
    end
    return (ranges..., (:len,) => 0:typemax(Int32))
end

# A device view's cached offset and stride are index-derived values. Leaving them wide
# would lose narrowing for strided views even when their parent and indices fit.
function GPUCompiler.specialization_ranges(::Type{S}) where {T,N,P<:CuDeviceArray,S<:SubArray{T,N,P}}
    ranges = ()
    for field in (:parent, :indices)
        for (path, range) in GPUCompiler.specialization_ranges(fieldtype(S, field))
            ranges = (ranges..., (field, path...) => range)
        end
    end
    return (ranges..., (:offset1,) => typemin(Int32):typemax(Int32),
                       (:stride1,) => typemin(Int32):typemax(Int32))
end
