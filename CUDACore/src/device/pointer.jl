# CUDA-specific operations on pointers with address spaces

## adrspace aliases

export AS

module AS

const Generic       = 0
const Global        = 1
const Shared        = 3
const Constant      = 4
const Local         = 5
const SharedCluster = 7

end


## ldg

const LDGTypes = (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64,
                  Float32, Float64)

# a load from addrspace(1) carrying `!invariant.load` metadata, which the NVPTX back-end
# lowers to `ld.global.nc`. A single method covers both scalar and vector element types,
# since `convert(LLVMType, T)` maps `NTuple{N, VecElement{T}}` to `<N x T>`.
@device_function @llvmgenerated builder function _pointerref_ldg(ptr::LLVMPtr{T,AS.Global},
                                                                 i::Int,
                                                                 ::Val{align})::T where {T, align}
    eltyp = convert(LLVMType, T)
    if supports_typed_pointers(LLVM.context())
        ptr = bitcast!(builder, ptr, LLVM.PointerType(eltyp, AS.Global))
    end
    ld = load!(builder, eltyp, inbounds_gep!(builder, eltyp, ptr, [i]); align)
    ld.metadata[MD_tbaa] = tbaa_addrspace(AS.Global)
    ld.metadata[MD_invariant_load] = MDNode(Metadata[])
    ld
end

@device_function @inline pointerref_ldg(ptr::LLVMPtr{T,AS.Global}, i::Int,
                                        align::Val) where {T} =
    _pointerref_ldg(ptr, i - 1, align)

for (N, T) in ((4, Float32), (2, Float64), (4, Int8), (4, Int16), (4, Int32), (2, Int64))
    @eval @inline unsafe_cached_load(p::LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, i::Integer=1, align::Val=Val(1)) =
        pointerref_ldg(p, Int(i), align)
end

# interface

export unsafe_cached_load

# Like `unsafe_load`, the index is widened to `Int` before reaching the intrinsic so that
# an unsigned index is zero-extended in Julia, rather than sign-extended by `getelementptr`.
@inline unsafe_cached_load(p::LLVMPtr{<:Union{LDGTypes...},AS.Global}, i::Integer=1, align::Val=Val(1)) =
    pointerref_ldg(p, Int(i), align)
# NOTE: fall back to normal unsafe_load for unsupported types. we could be smarter here,
#       e.g. destruct/load/reconstruct, but that's too complicated for what it's worth.
unsafe_cached_load(p::LLVMPtr, i::Integer=1, align::Val=Val(1)) =
    unsafe_load(p, i, align)
