# CUDA-specific operations on pointers with address spaces

## adrspace aliases

export AS

module AS

const Generic  = 0
const Global   = 1
const Shared   = 3
const Constant = 4
const Local    = 5

end


@inline @generated function pointerref_ldg(ptr::LLVMPtr{T,A}, i::I, ::Val{align}) where {T,A,I,align}
    sizeof(T) == 0 && return T.instance
    LLVM.@dispose ctx=LLVM.Context() begin
        eltyp = convert(LLVM.LLVMType, T)

        T_idx = convert(LLVM.LLVMType, I)
        T_ptr = convert(LLVM.LLVMType, ptr)

        T_typed_ptr = LLVM.PointerType(eltyp, A)

        # create a function
        param_types = [T_ptr, T_idx]
        llvm_f, _ = LLVM.Interop.create_function(eltyp, param_types)

        # generate IR
        LLVM.@dispose builder=LLVM.IRBuilder() begin
            entry = LLVM.BasicBlock(llvm_f, "entry")
            LLVM.position!(builder, entry)
            ptr = if LLVM.supports_typed_pointers(ctx)
                typed_ptr = LLVM.bitcast!(builder, LLVM.parameters(llvm_f)[1], T_typed_ptr)
                LLVM.inbounds_gep!(builder, eltyp, typed_ptr, [LLVM.parameters(llvm_f)[2]])
            else
                LLVM.inbounds_gep!(builder, eltyp, LLVM.parameters(llvm_f)[1], [LLVM.parameters(llvm_f)[2]])
            end
            ld = LLVM.load!(builder, eltyp, ptr)
            if A != 0
                LLVM.metadata(ld)[LLVM.MD_tbaa] = LLVM.Interop.tbaa_addrspace(A)
            end
            LLVM.alignment!(ld, align)
            LLVM.metadata(ld)[LLVM.MD_invariant_load] = LLVM.MDNode(LLVM.Metadata[nothing])

            LLVM.ret!(builder, ld)
        end

        LLVM.Interop.call_function(llvm_f, T, Tuple{LLVMPtr{T,A}, I}, :ptr, :(i-one(I)))
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
    @eval @inline function pointerref_ldg(base_ptr::LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, i::Integer,
                                          ::Val{align}) where align
        offset = i-one(i) # in elements
        ptr = base_ptr + offset*$N*sizeof($T)
        @typed_ccall($intr, llvmcall, $NTuple{$N, Base.VecElement{$T}}, (LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, Int32), ptr, Val(align))
    end
    @eval unsafe_cached_load(p::LLVMPtr{NTuple{$N, Base.VecElement{$T}},AS.Global}, i::Integer=1, align::Val=Val(1)) =
        pointerref_ldg(p, i, align)
end

# interface

export unsafe_cached_load

unsafe_cached_load(p::LLVMPtr{<:Union{LDGTypes...},AS.Global}, i::Integer=1, align::Val=Val(1)) =
    pointerref_ldg(p, i, align)
# NOTE: fall back to normal unsafe_load for unsupported types. we could be smarter here,
#       e.g. destruct/load/reconstruct, but that's too complicated for what it's worth.
unsafe_cached_load(p::LLVMPtr, i::Integer=1, align::Val=Val(1)) =
    unsafe_load(p, i, align)
