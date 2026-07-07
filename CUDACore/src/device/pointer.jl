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

# TODO: this functionality should throw <sm_32
# NOTE: CUDA 8.0 supports more caching modifiers, but those aren't supported by LLVM yet
@static if LLVM.version() >= v"20"
    # LLVM 20 removed the `llvm.nvvm.ldg.global.*` intrinsics in favor of a regular
    # load from addrspace(1) carrying `!invariant.load` metadata, which the NVPTX
    # backend lowers to `ld.global.nc` (see llvm/llvm-project#112834). We build that
    # IR with the LLVM.jl IRBuilder, mirroring `LLVM.Interop.pointerref`. A single
    # method covers both scalar and vector element types, since `convert(LLVMType, T)`
    # already maps `NTuple{N, VecElement{T}}` to `<N x T>`.
    @device_function @inline @generated function pointerref_ldg(ptr::LLVMPtr{T,AS.Global},
                                                                i::I, ::Val{align}) where {T, I, align}
        @dispose ctx=Context() begin
            eltyp = convert(LLVMType, T)
            T_idx = convert(LLVMType, I)
            T_ptr = convert(LLVMType, ptr)

            llvm_f, _ = create_function(eltyp, [T_ptr, T_idx])

            @dispose builder=IRBuilder() begin
                entry = BasicBlock(llvm_f, "entry")
                position!(builder, entry)
                gep = inbounds_gep!(builder, eltyp, parameters(llvm_f)[1],
                                    [parameters(llvm_f)[2]])
                ld = load!(builder, eltyp, gep)
                metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Global)
                metadata(ld)[LLVM.MD_invariant_load] = MDNode(Metadata[])
                alignment!(ld, align)
                ret!(builder, ld)
            end

            call_function(llvm_f, T, Tuple{LLVMPtr{T,AS.Global}, I}, :ptr, :(i-one(I)))
        end
    end

    for (N, T) in ((4, Float32), (2, Float64), (4, Int8), (4, Int16), (4, Int32), (2, Int64))
        @eval unsafe_cached_load(p::LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, i::Integer=1, align::Val=Val(1)) =
            pointerref_ldg(p, i, align)
    end
else
    for T in LDGTypes
        class = if T <: Integer
            :i
        elseif T <: AbstractFloat
            :f
        end
        # TODO: p class
        width = sizeof(T)*8 # in bits
        typ = Symbol(class, width)

        intr = "llvm.nvvm.ldg.global.$class.$typ.p1$typ"
        @eval @device_function @inline function pointerref_ldg(base_ptr::LLVMPtr{$T,AS.Global}, i::Integer,
                                              ::Val{align}) where align
            offset = i-one(i) # in elements
            ptr = base_ptr + offset*sizeof($T)
            @typed_ccall($intr, llvmcall, $T, (LLVMPtr{$T,AS.Global}, Int32), ptr, Val(align))
        end
    end

    for (N, T) in ((4, Float32), (2, Float64), (4, Int8), (4, Int16), (4, Int32), (2, Int64))
        class = if T <: Integer
            :i
        elseif T <: AbstractFloat
            :f
        end
        # TODO: p class
        width = sizeof(T)*8 # in bits
        typ = Symbol(class, width)

        intr = "llvm.nvvm.ldg.global.$class.v$N$typ.p1v$N$typ"
        @eval @device_function @inline function pointerref_ldg(base_ptr::LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, i::Integer,
                                              ::Val{align}) where align
            offset = i-one(i) # in elements
            ptr = base_ptr + offset*$N*sizeof($T)
            @typed_ccall($intr, llvmcall, $NTuple{$N, Base.VecElement{$T}}, (LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, Int32), ptr, Val(align))
        end
        @eval unsafe_cached_load(p::LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, i::Integer=1, align::Val=Val(1)) =
            pointerref_ldg(p, i, align)
    end
end

# interface

export unsafe_cached_load

unsafe_cached_load(p::LLVMPtr{<:Union{LDGTypes...},AS.Global}, i::Integer=1, align::Val=Val(1)) =
    pointerref_ldg(p, i, align)
# NOTE: fall back to normal unsafe_load for unsupported types. we could be smarter here,
#       e.g. destruct/load/reconstruct, but that's too complicated for what it's worth.
unsafe_cached_load(p::LLVMPtr, i::Integer=1, align::Val=Val(1)) =
    unsafe_load(p, i, align)
