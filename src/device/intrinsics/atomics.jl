# Atomic Functions (B.12)

#
# Low-level intrinsics
#

# TODO:
# - scoped atomics: _system and _block versions (see CUDA programming guide, sm_60+)
#   https://github.com/Microsoft/clang/blob/86d4513d3e0daa4d5a29b0b1de7c854ca15f9fe5/test/CodeGen/builtins-nvptx.c#L293

## LLVM

# common arithmetic operations on integers using LLVM instructions
#
# > 8.6.6. atomicrmw Instruction
# >
# > nand is not supported. The other keywords are supported for i32 and i64 types, with the
# > following restrictions.
# >
# > - The pointer must be either a global pointer, a shared pointer, or a generic pointer
# >   that points to either the global address space or the shared address space.

@generated function llvm_atomic_op(::Val{binop}, ptr::LLVMPtr{T,A}, val::T) where {binop, T, A}
    JuliaContext() do ctx
        T_val = convert(LLVMType, T, ctx)
        T_ptr = convert(LLVMType, ptr, ctx)

        T_typed_ptr = LLVM.PointerType(T_val, A)

        llvm_f, _ = create_function(T_val, [T_ptr, T_val])

        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = bitcast!(builder, parameters(llvm_f)[1], T_typed_ptr)

            rv = atomic_rmw!(builder, binop,
                            typed_ptr, parameters(llvm_f)[2],
                            atomic_acquire_release, #=single_threaded=# false)

            ret!(builder, rv)
        end

        call_function(llvm_f, T, Tuple{LLVMPtr{T,A}, T}, :((ptr,val)))
    end
end

const binops = Dict(
    :xchg  => LLVM.API.LLVMAtomicRMWBinOpXchg,
    :add   => LLVM.API.LLVMAtomicRMWBinOpAdd,
    :sub   => LLVM.API.LLVMAtomicRMWBinOpSub,
    :and   => LLVM.API.LLVMAtomicRMWBinOpAnd,
    :or    => LLVM.API.LLVMAtomicRMWBinOpOr,
    :xor   => LLVM.API.LLVMAtomicRMWBinOpXor,
    :max   => LLVM.API.LLVMAtomicRMWBinOpMax,
    :min   => LLVM.API.LLVMAtomicRMWBinOpMin,
    :umax  => LLVM.API.LLVMAtomicRMWBinOpUMax,
    :umin  => LLVM.API.LLVMAtomicRMWBinOpUMin,
    #:fadd  => LLVM.API.LLVMAtomicRMWBinOpFAdd,
    #:fsub  => LLVM.API.LLVMAtomicRMWBinOpFSub,
)

# all atomic operations have acquire and/or release semantics,
# depending on whether they load or store values (mimics Base)
const atomic_acquire = LLVM.API.LLVMAtomicOrderingAcquire
const atomic_release = LLVM.API.LLVMAtomicOrderingRelease
const atomic_acquire_release = LLVM.API.LLVMAtomicOrderingAcquireRelease

for T in (Int32, Int64, UInt32, UInt64)
    ops = [:xchg, :add, :sub, :and, :or, :xor, :max, :min]

    ptrs = LLVMPtr{}

    for op in ops
        # LLVM distinguishes signedness in the operation, not the integer type.
        rmw =  if T <: Unsigned && (op == :max || op == :min)
            Symbol("u$op")
        else
            Symbol("$op")
        end

        fn = Symbol("atomic_$(op)!")
        @eval @inline $fn(ptr::Union{LLVMPtr{$T,AS.Generic},
                                     LLVMPtr{$T,AS.Global},
                                     LLVMPtr{$T,AS.Shared}}, val::$T) =
            llvm_atomic_op($(Val(binops[rmw])), ptr, val)
    end
end

@generated function llvm_atomic_cas(ptr::LLVMPtr{T,A}, cmp::T, val::T) where {T, A}
    JuliaContext() do ctx
        T_val = convert(LLVMType, T, ctx)
        T_ptr = convert(LLVMType, ptr, ctx)

        T_typed_ptr = LLVM.PointerType(T_val, A)

        llvm_f, _ = create_function(T_val, [T_ptr, T_val, T_val])

        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = bitcast!(builder, parameters(llvm_f)[1], T_typed_ptr)

            res = atomic_cmpxchg!(builder, typed_ptr, parameters(llvm_f)[2],
                                parameters(llvm_f)[3], atomic_acquire_release, atomic_acquire,
                                #=single threaded=# false)

            rv = extract_value!(builder, res, 0)

            ret!(builder, rv)
        end

        call_function(llvm_f, T, Tuple{LLVMPtr{T,A}, T, T}, :((ptr,cmp,val)))
    end
end

for T in (Int32, Int64, UInt32, UInt64)
    @eval @inline atomic_cas!(ptr::LLVMPtr{$T}, cmp::$T, val::$T) =
        llvm_atomic_cas(ptr, cmp, val)
end


## NVVM

# floating-point operations using NVVM intrinsics

for A in (AS.Generic, AS.Global, AS.Shared)
    # declare float @llvm.nvvm.atomic.load.add.f32.p0f32(float* address, float val)
    # declare float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* address, float val)
    # declare float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* address, float val)
    #
    # FIXME: these only works on sm_60+, but we can't verify that for now
    # declare double @llvm.nvvm.atomic.load.add.f64.p0f64(double* address, double val)
    # declare double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)* address, double val)
    # declare double @llvm.nvvm.atomic.load.add.f64.p3f64(double addrspace(3)* address, double val)
    for T in (Float32, Float64)
        nb = sizeof(T)*8
        intr = "llvm.nvvm.atomic.load.add.f$nb.p$(convert(Int, A))f$nb"
        #@eval @inline atomic_add!(ptr::LLVMPtr{$T,$A}, val::$T) =
        #    @typed_ccall($intr, llvmcall, $T, (LLVMPtr{$T,$A}, $T), ptr, val)
        # NOTE: this is implemented as `atomicrmw fadd` on LLVM 9+, but the C headers
        #       lack an export: https://github.com/llvm/llvm-project/commit/f57e968dd036b2230c59c00e1ed10fecf1668828
        #       by using IR, we trigger the AutoUpgrader which fixes this.
        # TODO: implement as atomicrmw when using LLVM 10 (Julia 1.6)

        import Base.Sys: WORD_SIZE
        if T == Float32
            T_val = "float"
        else
            T_val = "double"
        end
        if A == AS.Generic
            T_untyped_ptr = "i8*"
            T_typed_ptr = "$(T_val)*"
        else
            T_untyped_ptr = "i8 addrspace($(convert(Int, A)))*"
            T_typed_ptr = "$(T_val) addrspace($(convert(Int, A)))*"
        end
        mod = """
            declare $T_val @$intr($T_typed_ptr, $T_val)

            define $T_val @entry($T_untyped_ptr %0, $T_val %1) #0 {
                %ptr = bitcast $T_untyped_ptr %0 to $T_typed_ptr
                %rv = call $T_val @$intr($T_typed_ptr %ptr, $T_val %1)
                ret $T_val %rv
            }

            attributes #0 = { alwaysinline }"""
        @eval @inline atomic_add!(ptr::LLVMPtr{$T,$A}, val::$T) =
            Base.llvmcall(($mod, "entry"), $T, Tuple{LLVMPtr{$T,$A}, $T}, ptr, val)
    end

    # declare i32 @llvm.nvvm.atomic.load.inc.32.p0i32(i32* address, i32 val)
    # declare i32 @llvm.nvvm.atomic.load.inc.32.p1i32(i32 addrspace(1)* address, i32 val)
    # declare i32 @llvm.nvvm.atomic.load.inc.32.p3i32(i32 addrspace(3)* address, i32 val)
    #
    # declare i32 @llvm.nvvm.atomic.load.dec.32.p0i32(i32* address, i32 val)
    # declare i32 @llvm.nvvm.atomic.load.dec.32.p1i32(i32 addrspace(1)* address, i32 val)
    # declare i32 @llvm.nvvm.atomic.load.dec.32.p3i32(i32 addrspace(3)* address, i32 val)
    for T in (Int32,), op in (:inc, :dec)
        nb = sizeof(T)*8
        fn = Symbol("atomic_$(op)!")
        intr = "llvm.nvvm.atomic.load.$op.$nb.p$(convert(Int, A))i$nb"
        @eval @inline $fn(ptr::LLVMPtr{$T,$A}, val::$T) =
            @typed_ccall($intr, llvmcall, $T, (LLVMPtr{$T,$A}, $T), ptr, val)
    end
end

# half-precision atomics using PTX instruction 

for A in (AS.Generic, AS.Global, AS.Shared)

    if A == AS.Global
        scope = ".global"
    elseif A == AS.Shared
        scope = ".shared"
    else
        scope = ""
    end

    T = Symbol(Float16)
    PTX = "atom$scope.add.noftz.f16 \$0, [\$1], \$2;"

    @eval @inline atomic_add!(ptr::LLVMPtr{$T,$A}, val::$T) =
        @Interop.asmcall($PTX, "=h,l,h", true, $T, Tuple{Core.LLVMPtr{$T,$A},$T}, ptr, val) 
end

## Julia

# floating-point CAS via bitcasting

inttype(::Type{T}) where {T<:Integer} = T
inttype(::Type{Float16}) = Int16
inttype(::Type{Float32}) = Int32
inttype(::Type{Float64}) = Int64

for T in [Float32, Float64]
    @eval @inline function atomic_cas!(ptr::LLVMPtr{$T,A}, cmp::$T, new::$T) where {A}
        IT = inttype($T)
        cmp_i = reinterpret(IT, cmp)
        new_i = reinterpret(IT, new)
        old_i = atomic_cas!(reinterpret(LLVMPtr{IT,A}, ptr), cmp_i, new_i)
        return reinterpret($T, old_i)
    end
end

# floating-point operations via atomic_cas!

const opnames = Dict{Symbol, Symbol}(:- => :sub, :* => :mul, :/ => :div)

for T in [Float32, Float64]
    for op in [:-, :*, :/, :max, :min]
        opname = get(opnames, op, op)
        fn = Symbol("atomic_$(opname)!")
        @eval @inline function $fn(ptr::LLVMPtr{$T}, val::$T)
            old = Base.unsafe_load(ptr, 1)
            while true
                cmp = old
                new = $op(old, val)
                old = atomic_cas!(ptr, cmp, new)
                (old == cmp) && return new
            end
        end
    end
end


## documentation

"""
    atomic_cas!(ptr::LLVMPtr{T}, cmp::T, val::T)

Reads the value `old` located at address `ptr` and compare with `cmp`. If `old` equals to
`cmp`, stores `val` at the same address. Otherwise, doesn't change the value `old`. These
operations are performed in one atomic transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
"""
atomic_cas!

"""
    atomic_xchg!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr` and stores `val` at the same address. These
operations are performed in one atomic transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
"""
atomic_xchg!

"""
    atomic_add!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `old + val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32, UInt64, and Float32.
Additionally, on GPU hardware with compute capability 6.0+, values of type Float64 are
supported.
"""
atomic_add!

"""
    atomic_sub!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `old - val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Additionally, on GPU hardware with compute capability 6.0+, values of type Float32 and
Float64 are supported.
"""
atomic_sub!

"""
    atomic_mul!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `*(old, val)`, and stores the
result back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported on GPU hardware with compute capability 6.0+ for values of type
Float32 and Float64.
"""
atomic_mul!

"""
    atomic_div!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `/(old, val)`, and stores the
result back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported on GPU hardware with compute capability 6.0+ for values of type
Float32 and Float64.
"""
atomic_div!

"""
    atomic_and!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `old & val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
"""
atomic_and!

"""
    atomic_or!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `old | val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
"""
atomic_or!

"""
    atomic_xor!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `old ⊻ val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
"""
atomic_xor!

"""
    atomic_min!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `min(old, val)`, and stores the
result back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Additionally, on GPU hardware with compute capability 6.0+, values of type Float32 and
Float64 are supported.
"""
atomic_min!

"""
    atomic_max!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `max(old, val)`, and stores the
result back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Additionally, on GPU hardware with compute capability 6.0+, values of type Float32 and
Float64 are supported.
"""
atomic_max!

"""
    atomic_inc!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `((old >= val) ? 0 : (old+1))`, and
stores the result back to memory at the same address. These three operations are performed
in one atomic transaction. The function returns `old`.

This operation is only supported for values of type Int32.
"""
atomic_inc!

"""
    atomic_dec!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `(((old == 0) | (old > val)) ? val
: (old-1) )`, and stores the result back to memory at the same address. These three
operations are performed in one atomic transaction. The function returns `old`.

This operation is only supported for values of type Int32.
"""
atomic_dec!



#
# High-level interface
#

# prototype of a high-level interface for performing atomic operations on arrays
#
# this design could be generalized by having atomic {field,array}{set,ref} accessors, as
# well as acquire/release operations to implement the fallback functionality where any
# operation can be applied atomically.

export @atomic

const inplace_ops = Dict(
    :(+=) => :(+),
    :(-=) => :(-),
    :(*=) => :(*),
    :(/=) => :(/),
    :(&=) => :(&),
    :(|=) => :(|),
    :(⊻=) => :(⊻)
)

struct AtomicError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, err::AtomicError) =
    print(io, "AtomicError: ", err.msg)

"""
    @atomic a[I] = op(a[I], val)
    @atomic a[I] ...= val

Atomically perform a sequence of operations that loads an array element `a[I]`, performs the
operation `op` on that value and a second value `val`, and writes the result back to the
array. This sequence can be written out as a regular assignment, in which case the same
array element should be used in the left and right hand side of the assignment, or as an
in-place application of a known operator. In both cases, the array reference should be pure
and not induce any side-effects.

!!! warn
    This interface is experimental, and might change without warning.  Use the lower-level
    `atomic_...!` functions for a stable API.
"""
macro atomic(ex)
    # decode assignment and call
    if ex.head == :(=)
        ref = ex.args[1]
        rhs = ex.args[2]
        Meta.isexpr(rhs, :call) || throw(AtomicError("right-hand side of an @atomic assignment should be a call"))
        op = rhs.args[1]
        if rhs.args[2] != ref
            throw(AtomicError("right-hand side of a non-inplace @atomic assignment should reference the left-hand side"))
        end
        val = rhs.args[3]
    elseif haskey(inplace_ops, ex.head)
        op = inplace_ops[ex.head]
        ref = ex.args[1]
        val = ex.args[2]
    else
        throw(AtomicError("unknown @atomic expression"))
    end

    # decode array expression
    Meta.isexpr(ref, :ref) || throw(AtomicError("@atomic should be applied to an array reference expression"))
    array = ref.args[1]
    indices = Expr(:tuple, ref.args[2:end]...)

    esc(quote
        $atomic_arrayset($array, $indices, $op, $val)
    end)
end

# FIXME: make this respect the indexing style
@inline atomic_arrayset(A::AbstractArray, Is::Tuple, op::Function, val) =
    atomic_arrayset(A, Base._to_linear_index(A, Is...), op, val)

function atomic_arrayset(A::AbstractArray, I::Integer, op::Function, val)
    error("Don't know how to atomically perform $op on $(typeof(A))")
    # TODO: while { acquire, op, cmpxchg }
end

# CUDA.jl atomics
for (op,impl) in [(+)      => atomic_add!,
                  (-)      => atomic_sub!,
                  (*)      => atomic_mul!,
                  (/)      => atomic_div!,
                  (&)      => atomic_and!,
                  (|)      => atomic_or!,
                  (⊻)      => atomic_xor!,
                  Base.max => atomic_max!,
                  Base.min => atomic_min!]
    @eval @inline atomic_arrayset(A::CuDeviceArray, I::Integer, ::typeof($op), val) =
        $impl(pointer(A, I), val)
end
