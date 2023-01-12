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


## ldg

const LDGTypes = (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64,
                  Float32, Float64)

# TODO: this functionality should throw <sm_32
# NOTE: CUDA 8.0 supports more caching modifiers, but those aren't supported by LLVM yet
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
    @eval @inline function pointerref_ldg(base_ptr::LLVMPtr{$T,AS.Global}, i::Integer,
                                          ::Val{align}) where align
        offset = i-one(i) # in elements
        ptr = base_ptr + offset*sizeof($T)
        @typed_ccall($intr, llvmcall, $T, (LLVMPtr{$T,AS.Global}, Int32), ptr, Val(align))
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
