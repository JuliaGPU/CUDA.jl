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

# interface

export unsafe_cached_load

# Like `unsafe_load`, the index is widened to `Int` before reaching the intrinsic so that
# an unsigned index is zero-extended in Julia, rather than sign-extended by `getelementptr`.
@inline unsafe_cached_load(p::LLVMPtr{<:Union{LDGTypes...},AS.Global}, i::Integer=1, align::Val=Val(1)) =
    pointerref_ldg(p, Int(i), align)
# other element types, e.g. `ComplexF32` or `NTuple{4,Float32}`. LLVM drops `!invariant.load`
# when it splits a load of such an aggregate into loads of its fields, or a vector load into
# loads of its elements, so `pointerref_ldg` wouldn't reliably result in `ld.global.nc`.
# like CUDA C's `__ldg` for vector types, use inline assembly for elements that can be
# loaded at once. that matters before Volta, where `ld.global.nc` uses a separate cache.
@inline function unsafe_cached_load(p::LLVMPtr{T,AS.Global}, i::Integer=1,
                                    ::Val{align}=Val(1)) where {T,align}
    if isbitstype(T) && sizeof(T) in (4, 8, 16) && align >= sizeof(T)
        q = p + (Int(i)-1) * aligned_sizeof(T)
        reinterpret_bits(T, ldg_bits(Val(sizeof(T)), UInt64(UInt(q))))
    elseif isbitstype(T) && sizeof(T) > 0
        pointerref_ldg(p, Int(i), Val(align))
    else
        unsafe_load(p, i, Val(align))
    end
end

@device_function ldg_bits(::Val{4}, addr::UInt64) = Base.llvmcall(
    """%v = call i32 asm "ld.global.nc.b32 \$0, [\$1];", "=r,l"(i64 %0)
       ret i32 %v""", UInt32, Tuple{UInt64}, addr)
@device_function ldg_bits(::Val{8}, addr::UInt64) = Base.llvmcall(
    """%v = call i64 asm "ld.global.nc.b64 \$0, [\$1];", "=l,l"(i64 %0)
       ret i64 %v""", UInt64, Tuple{UInt64}, addr)
@device_function ldg_bits(::Val{16}, addr::UInt64) = Base.llvmcall(
    """%v = call { i64, i64 } asm "ld.global.nc.v2.b64 {\$0, \$1}, [\$2];", "=l,=l,l"(i64 %0)
       %a = extractvalue { i64, i64 } %v, 0
       %b = extractvalue { i64, i64 } %v, 1
       %r0 = insertelement <2 x i64> poison, i64 %a, i32 0
       %r = insertelement <2 x i64> %r0, i64 %b, i32 1
       ret <2 x i64> %r""", NTuple{2,VecElement{UInt64}}, Tuple{UInt64}, addr)

# NOTE: fall back to normal unsafe_load for unsupported types
unsafe_cached_load(p::LLVMPtr, i::Integer=1, align::Val=Val(1)) =
    unsafe_load(p, i, align)
