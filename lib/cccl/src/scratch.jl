# scratch memory
#
# Every algorithm needs CUB's temporary storage, whose size only the build of the algorithm
# can tell, and some need buffers of their own (e.g. the second half of a radix sort's
# double buffer). They are laid out in one block of device memory, which the algorithm
# allocates or takes from a user-provided `workspace`, so that callers can allocate the
# memory up front (see `workspace_size`).

# offsets of the buffers within the block, which suits CUB's temporary storage too
const scratch_alignment = 256

aligned(nbytes::Integer) = cld(nbytes, scratch_alignment) * scratch_alignment

# a buffer of `length` elements of type `T`
struct Buffer{T}
    length::Int
end
nbytes(b::Buffer{T}) where {T} = b.length * sizeof(T)

# passed as `workspace` to compute the size of the scratch memory instead of running
struct SizeQuery end

# the temporary storage a CUB-style function `f(temp, nbytes)` needs, which it reports when
# called without any. That is at least a byte, since a null pointer means a query.
function temp_size(f)
    nbytes = Ref{Csize_t}(0)
    f(C_NULL, nbytes)
    return max(Int(nbytes[]), 1)
end

function check_workspace(workspace::DenseCuVector{UInt8}, size::Int)
    length(workspace) >= size || throw(ArgumentError(
        "workspace holds $(length(workspace)) bytes, but this call needs $size; see CCCL.workspace_size"))
    UInt(pointer(workspace)) % scratch_alignment == 0 || throw(ArgumentError(
        "workspace must be aligned to $scratch_alignment bytes, as freshly allocated arrays are"))
    return
end

"""
    with_scratch(f, workspace, temp, buffers::Buffer...)

Call `f(temp_ptr, arrays...)` with a pointer to `temp` bytes of temporary storage and arrays
for each of the `buffers`, all taken from `workspace` (or allocated if it is `nothing`), and
return its result. With `workspace = SizeQuery()`, instead return the number of bytes needed.
"""
function with_scratch(f, workspace, temp::Int, buffers::Buffer...)
    sizes = map(aligned ∘ nbytes, buffers)
    total = sum(sizes; init=0) + temp
    workspace isa SizeQuery && return total

    memory = if workspace === nothing
        CuVector{UInt8}(undef, total)
    else
        check_workspace(workspace, total)
        workspace
    end
    try
        GC.@preserve memory begin
            base = pointer(memory)
            offsets = cumsum((0, sizes...))
            arrays = map(buffers, offsets[1:end-1]) do b, offset
                wrap(memory, b, base + offset)
            end
            temp_ptr = temp == 0 ? C_NULL : reinterpret(Ptr{Cvoid}, base + offsets[end])
            f(temp_ptr, arrays...)
        end
    finally
        workspace === nothing && CUDACore.unsafe_free!(memory)
    end
end

# the arrays do not own their memory; `with_scratch` keeps it alive while they are used
wrap(::CuVector{UInt8,M}, b::Buffer{T}, ptr) where {M,T} =
    unsafe_wrap(CuArray{T,1,M}, reinterpret(CuPtr{T}, ptr), b.length)
