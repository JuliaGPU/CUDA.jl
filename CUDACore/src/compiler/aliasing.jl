# launch-time alias information for kernel arguments
#
# At launch, the converted arguments of a kernel tell exactly which memory every device
# array in them covers. Arrays that cover disjoint memory get different alias classes,
# which GPUCompiler turns into scoped `noalias` metadata on the accesses the kernel makes
# through them. The classes are encoded in an `alias_key` that is part of the compiler
# params, so kernels are compiled once for every aliasing pattern they are launched with.

"""
    CUDACore.ARGUMENT_ALIASING

Whether to specialize kernels on which of their array arguments alias each other.
"""
const ARGUMENT_ALIASING = Ref(true)

"""
    CUDACore.ARGUMENT_INVARIANT_LOADS

Whether kernels that are specialized on argument aliasing load arrays they don't write to
through the non-coherent cache (`ld.global.nc`).
"""
const ARGUMENT_INVARIANT_LOADS = Ref(false)

# the device arrays in the global address space that are reachable through the fields of
# an argument of type `T`: their field path and the byte offset of their pointer.
function alias_leaves!(leaves, @nospecialize(T), path, offset)
    T isa DataType || return leaves
    if T <: CuDeviceArray
        if T.parameters[3] == AS.Global
            push!(leaves, (path, offset + fieldoffset(T, 1)))
        end
    elseif isbitstype(T) && !(T <: Core.LLVMPtr) && !(T <: Ptr)
        for i in 1:fieldcount(T)
            alias_leaves!(leaves, fieldtype(T, i), (path..., i), offset + Int(fieldoffset(T, i)))
        end
    end
    return leaves
end

# the leaves of every argument in a signature `Tuple{F, args...}`, as `(arg, path, offset)`
function alias_leaves(@nospecialize(sig))
    leaves = Tuple{Int,Tuple,Int}[]
    for (i, T) in enumerate(sig.parameters)
        for (path, offset) in alias_leaves!([], T, (), 0)
            push!(leaves, (i, path, offset))
        end
    end
    return leaves
end

# the bytes a device array covers, including the selector bytes of isbits-union arrays
@inline function alias_extent(a::CuDeviceArray{T}) where {T}
    start = UInt(a.ptr)
    nbytes = Base.isbitsunion(T) ? a.maxsize + a.len : a.len * aligned_sizeof(T)
    return start, start + UInt(nbytes)
end

const MAX_ALIAS_LEAVES = 16

# partition arrays into classes of overlapping ones, numbered by first appearance and
# packed in 4 bits per array. returns 0 when all arrays are in a single class.
@inline function alias_partition(starts::NTuple{L,UInt}, stops::NTuple{L,UInt}) where {L}
    labels = ntuple(identity, Val(L))
    for i in 1:L, j in i+1:L
        si, ei, sj, ej = starts[i], stops[i], starts[j], stops[j]
        (si < ei && sj < ej && si < ej && sj < ei) || continue
        li, lj = labels[i], labels[j]
        li == lj && continue
        labels = map(l -> l == lj ? li : l, labels)
    end
    key = UInt64(0)
    nclasses = 0
    seen = ntuple(_ -> 0, Val(L))    # label => class
    for i in 1:L
        l = labels[i]
        if seen[l] == 0
            nclasses += 1
            seen = Base.setindex(seen, nclasses, l)
        end
        key |= UInt64(seen[l] - 1) << (4 * (i - 1))
    end
    return nclasses >= 2 ? key : UInt64(0)
end

"""
    alias_key(f, args::Tuple) -> UInt64

The aliasing pattern of the device arrays in converted kernel arguments: the class of every
array, packed in 4 bits per array in the order of `alias_leaves`, or 0 for no information.
"""
@generated function alias_key(f, args::Tuple)
    sig = Tuple{f, args.parameters...}
    leaves = alias_leaves(sig)
    L = length(leaves)
    2 <= L <= MAX_ALIAS_LEAVES || return :(UInt64(0))
    extents = map(leaves) do (i, path, _)
        ex = i == 1 ? :f : :(args[$(i - 1)])
        for field in path
            ex = :(getfield($ex, $field))
        end
        :(alias_extent($ex))
    end
    quote
        ARGUMENT_ALIASING[] || return UInt64(0)
        extents = ($(extents...),)
        alias_partition(map(first, extents), map(last, extents))
    end
end

function GPUCompiler.kernel_argument_alias_classes(@nospecialize(job::AnyCUDAJob))
    key = job.config.params.alias_key
    key == 0 && return nothing
    leaves = alias_leaves(job.source.specTypes)
    return [(arg, offset, Int((key >> (4 * (j - 1))) & 0xf) + 1)
            for (j, (arg, _, offset)) in enumerate(leaves)]
end

GPUCompiler.kernel_argument_invariant_loads(@nospecialize(job::AnyCUDAJob)) =
    job.config.params.alias_invariant
