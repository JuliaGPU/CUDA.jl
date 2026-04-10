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
    # backend lowers to `ld.global.nc` (see llvm/llvm-project#112834).

    function ldg_llvm_type(@nospecialize(T::Type))
        T === UInt8  || T === Int8    ? "i8"     :
        T === UInt16 || T === Int16   ? "i16"    :
        T === UInt32 || T === Int32   ? "i32"    :
        T === UInt64 || T === Int64   ? "i64"    :
        T === Float32                 ? "float"  :
        T === Float64                 ? "double" :
        error("Unsupported LDG type: $T")
    end

    @device_function @generated function pointerref_ldg(base_ptr::LLVMPtr{T,AS.Global},
                                                        i::Integer, ::Val{align}) where {T, align}
        llvm_typ = ldg_llvm_type(T)
        ir = """
            define $llvm_typ @entry(ptr addrspace(1) %ptr) #0 {
                %v = load $llvm_typ, ptr addrspace(1) %ptr, align $align, !invariant.load !0
                ret $llvm_typ %v
            }
            attributes #0 = { alwaysinline }
            !0 = !{}
        """
        quote
            offset = i-one(i) # in elements
            ptr = base_ptr + offset*sizeof(T)
            Base.llvmcall(($ir, "entry"), T, Tuple{LLVMPtr{T,AS.Global}}, ptr)
        end
    end

    @device_function @generated function pointerref_ldg(
            base_ptr::LLVMPtr{NTuple{N, Base.VecElement{T}},AS.Global},
            i::Integer, ::Val{align}) where {N, T, align}
        llvm_typ = "<$N x $(ldg_llvm_type(T))>"
        ir = """
            define $llvm_typ @entry(ptr addrspace(1) %ptr) #0 {
                %v = load $llvm_typ, ptr addrspace(1) %ptr, align $align, !invariant.load !0
                ret $llvm_typ %v
            }
            attributes #0 = { alwaysinline }
            !0 = !{}
        """
        quote
            offset = i-one(i) # in elements
            ptr = base_ptr + offset*N*sizeof(T)
            Base.llvmcall(($ir, "entry"), NTuple{N, Base.VecElement{T}},
                          Tuple{LLVMPtr{NTuple{N, Base.VecElement{T}},AS.Global}}, ptr)
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
