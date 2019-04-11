# Atomic Functions (B.12)

#
# Low-level intrinsics
#

# TODO:
# - scoped atomics: _system and _block versions (see CUDA programming guide, sm_60+)
#   https://github.com/Microsoft/clang/blob/86d4513d3e0daa4d5a29b0b1de7c854ca15f9fe5/test/CodeGen/builtins-nvptx.c#L293
# - atomic_cas!

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

@generated function llvm_atomic(::Val{binop}, ptr::DevicePtr{T,A}, val::T, ::Val{ordering}) where
                               {binop, T, A, ordering}
    T_val = convert(LLVMType, T)
    T_ptr = convert(LLVMType, DevicePtr{T,A})
    T_actual_ptr = LLVM.PointerType(T_val)

    llvm_f, _ = create_function(T_val, [T_ptr, T_val])

    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        actual_ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)

        rv = atomic_rmw!(builder, binop,
                         actual_ptr, parameters(llvm_f)[2],
                         ordering, #=single_threaded=# false)

        ret!(builder, rv)
    end

    call_function(llvm_f, T, Tuple{DevicePtr{T,A}, T}, :((ptr,val)))
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
    :umin  => LLVM.API.LLVMAtomicRMWBinOpUMin
)

# all atomic operations have acquire and/or release semantics,
# depending on whether they load or store values (mimics Base)
const aquire = LLVM.API.LLVMAtomicOrderingAcquire
const aquire_release = LLVM.API.LLVMAtomicOrderingAcquireRelease

for T in (Int32, Int64, UInt32, UInt64)
    ops = [:xchg, :add, :sub, :and, :or, :xor, :max, :min]

    ASs = Union{AS.Generic, AS.Global, AS.Shared}

    for op in ops
        # LLVM distinguishes signedness in the operation, not the integer type.
        rmw =  if T <: Unsigned && (op == :max || op == :min)
            Symbol("u$op")
        else
            Symbol("$op")
        end

        fn = Symbol("atomic_$(op)!")
        @eval @inline $fn(ptr::DevicePtr{$T,<:$ASs}, val::$T) =
            llvm_atomic($(Val(binops[rmw])), ptr, val, Val(aquire_release))
    end
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

        if VERSION >= v"999" # FIXME: JuliaLang/julia#31624
            @eval @inline atomic_add!(ptr::DevicePtr{$T,$A}, val::$T) =
                ccall($intr, llvmcall, $T, (DevicePtr{$T,$A}, $T), ptr, val)
        else
            import Base.Sys: WORD_SIZE
            if T == Float32
                T_val = "float"
            else
                T_val = "double"
            end
            if A == AS.Generic
                T_ptr = "$(T_val)*"
            else
                T_ptr = "$(T_val) addrspace($(convert(Int, A)))*"
            end
            @eval @inline atomic_add!(ptr::DevicePtr{$T,$A}, val::$T) = Base.llvmcall(
                $("declare $T_val @$intr($T_ptr, $T_val)",
                  "%ptr = inttoptr i$WORD_SIZE %0 to $T_ptr
                   %rv = call $T_val @$intr($T_ptr %ptr, $T_val %1)
                   ret $T_val %rv"), $T,
                Tuple{DevicePtr{$T,$A}, $T}, ptr, val)
        end
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
        intr = "llvm.nvvm.atomic.load.$op.$nb.p$(convert(Int, A))i$nb"
        fn = Symbol("atomic_$(op)!")

        if VERSION >= v"999" # FIXME: JuliaLang/julia#31624
            @eval @inline $fn(ptr::DevicePtr{$T,$A}, val::$T) =
                ccall($intr, llvmcall, $T, (DevicePtr{$T,$A}, $T), ptr, val)
        else
            import Base.Sys: WORD_SIZE
            T_val = "i32"
            if A == AS.Generic
                T_ptr = "$(T_val)*"
            else
                T_ptr = "$(T_val) addrspace($(convert(Int, A)))*"
            end
            @eval @inline $fn(ptr::DevicePtr{$T,$A}, val::$T) = Base.llvmcall(
                $("declare $T_val @$intr($T_ptr, $T_val)",
                  "%ptr = inttoptr i$WORD_SIZE %0 to $T_ptr
                   %rv = call $T_val @$intr($T_ptr %ptr, $T_val %1)
                   ret $T_val %rv"), $T,
                Tuple{DevicePtr{$T,$A}, $T}, ptr, val)
        end
    end
end


## documentation

"""
    atomic_add!(ptr::DevicePtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `old+val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32, UInt64, and Float32.
Additionally, on GPU hardware with compute capability 6.0+, values of type Float64 are
supported.
"""
atomic_add!

"""
    atomic_inc!(ptr::DevicePtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `((old >= val) ? 0 : (old+1))`, and
stores the result back to memory at the same address. These three operations are performed
in one atomic transaction. The function returns `old`.

This operation is only supported for values of type Int32.
"""
atomic_inc!

"""
    atomic_dec!(ptr::DevicePtr{T}, val::T)

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
    :(&=) => :(&),
    :(|=) => :(|),
    :(⊻=) => :(⊻)
)

"""
    @atomic a[I] = op(a[I], val)
    @atomic a[I] ...= val

Atomically perform a sequence of operations that loads an array element `a[I]`, performs
the operation `op` on that value and a second value `val`, and writes the result back to
the array. This sequence can be written out as a regular assignment, in which case the
same array element should be used in the left and right hand side of the assignment, or
as an in-place application of a known operator.
"""
macro atomic(ex)
    # decode assignment and call
    if ex.head == :(=)
        ref = ex.args[1]
        rhs = ex.args[2]
        rhs.head == :call || error("right-hand side of an @atomic assignment should be a call")
        op = rhs.args[1]
        if rhs.args[2] != ref
            error("non-inplace @atomic assignment should reference the same array elements")
        end
        val = rhs.args[3]
    elseif haskey(inplace_ops, ex.head)
        op = inplace_ops[ex.head]
        ref = ex.args[1]
        val = ex.args[2]
    else
        error("unknown @atomic expression")
    end

    # decode array expression
    if ref.head != :ref
        error("@atomic should be applied to an array reference expression")
    end
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

# CUDAnative.jl atomics
for (op,impl) in [(+)      => atomic_add!,
                  (-)      => atomic_sub!,
                  (&)      => atomic_and!,
                  (|)      => atomic_or!,
                  (⊻)      => atomic_xor!,
                  Base.max => atomic_max!,
                  Base.min => atomic_min!]
    @eval @inline atomic_arrayset(A::CuDeviceArray, I::Integer, ::typeof($op), val) =
        $impl(pointer(A, I), val)
end
