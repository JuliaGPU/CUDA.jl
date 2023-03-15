# Atomic Functions (B.12)

# TODO replace the below with UnsafeAtomicsLLVM if possible

#
# Low-level intrinsics
#

# TODO:
# - scoped atomics: _system and _block versions (see CUDA programming guide, sm_60+)
#   https://github.com/Microsoft/clang/blob/86d4513d3e0daa4d5a29b0b1de7c854ca15f9fe5/test/CodeGen/builtins-nvptx.c#L293

## LLVM

# all atomic operations have acquire and/or release semantics,
# depending on whether they load or store values (mimics Base)
const atomic_acquire = LLVM.API.LLVMAtomicOrderingAcquire
const atomic_release = LLVM.API.LLVMAtomicOrderingRelease
const atomic_acquire_release = LLVM.API.LLVMAtomicOrderingAcquireRelease

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
    @dispose ctx=Context() begin
        T_val = convert(LLVMType, T; ctx)
        T_ptr = convert(LLVMType, ptr; ctx)

        T_typed_ptr = LLVM.PointerType(T_val, A)

        llvm_f, _ = create_function(T_val, [T_ptr, T_val])

        @dispose builder=Builder(ctx) begin
            entry = BasicBlock(llvm_f, "entry"; ctx)
            position!(builder, entry)

            typed_ptr = bitcast!(builder, parameters(llvm_f)[1], T_typed_ptr)

            rv = atomic_rmw!(builder, binop,
                            typed_ptr, parameters(llvm_f)[2],
                            atomic_acquire_release, #=single_threaded=# false)

            ret!(builder, rv)
        end

        call_function(llvm_f, T, Tuple{LLVMPtr{T,A}, T}, :ptr, :val)
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
    :fadd  => LLVM.API.LLVMAtomicRMWBinOpFAdd,
    :fsub  => LLVM.API.LLVMAtomicRMWBinOpFSub,
)

for T in (Int32, Int64, UInt32, UInt64)
    ops = [:xchg, :add, :sub, :and, :or, :xor, :max, :min]

    for op in ops
        # LLVM distinguishes signedness in the operation, not the integer type.
        rmw = if T <: Unsigned && (op == :max || op == :min)
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

for T in (Float32, Float64)
    ops = [:add]

    for op in ops
        # LLVM has specific operations for floating point types.
        rmw = Symbol("f$op")

        fn = Symbol("atomic_$(op)!")
        # XXX: cannot select
        @eval @inline $fn(ptr::Union{LLVMPtr{$T,AS.Generic},
                                     LLVMPtr{$T,AS.Global},
                                     LLVMPtr{$T,AS.Shared}}, val::$T) =
           llvm_atomic_op($(Val(binops[rmw])), ptr, val)
    end

    # there's no specific NNVM intrinsic for fsub, resulting in a selection error.
    @eval @inline atomic_sub!(ptr::Union{LLVMPtr{$T,AS.Generic},
                                         LLVMPtr{$T,AS.Global},
                                         LLVMPtr{$T,AS.Shared}}, val::$T) =
        atomic_add!(ptr, -val)
end

@generated function llvm_atomic_cas(ptr::LLVMPtr{T,A}, cmp::T, val::T) where {T, A}
    @dispose ctx=Context() begin
        T_val = convert(LLVMType, T; ctx)
        T_ptr = convert(LLVMType, ptr,;ctx)

        T_typed_ptr = LLVM.PointerType(T_val, A)

        llvm_f, _ = create_function(T_val, [T_ptr, T_val, T_val])

        @dispose builder=Builder(ctx) begin
            entry = BasicBlock(llvm_f, "entry"; ctx)
            position!(builder, entry)

            typed_ptr = bitcast!(builder, parameters(llvm_f)[1], T_typed_ptr)

            res = atomic_cmpxchg!(builder, typed_ptr, parameters(llvm_f)[2],
                                parameters(llvm_f)[3], atomic_acquire_release, atomic_acquire,
                                #=single threaded=# false)

            rv = extract_value!(builder, res, 0)

            ret!(builder, rv)
        end

        call_function(llvm_f, T, Tuple{LLVMPtr{T,A}, T, T}, :ptr, :cmp, :val)
    end
end

for T in (Int32, Int64, UInt32, UInt64)
    @eval @inline atomic_cas!(ptr::LLVMPtr{$T}, cmp::$T, val::$T) =
        llvm_atomic_cas(ptr, cmp, val)
end

# NVPTX doesn't support cmpxchg with i16 yet
for A in (AS.Generic, AS.Global, AS.Shared), T in (Int16, UInt16)
    if A == AS.Global
        scope = ".global"
    elseif A == AS.Shared
        scope = ".shared"
    else
        scope = ""
    end

    intr = "atom$scope.cas.b16 \$0, [\$1], \$2, \$3;"
    @eval @inline atomic_cas!(ptr::LLVMPtr{$T,$A}, cmp::$T, val::$T) =
        @asmcall($intr, "=h,l,h,h", true, $T, Tuple{Core.LLVMPtr{$T,$A},$T,$T}, ptr, cmp, val)
end


## NVVM

# floating-point operations using NVVM intrinsics

for A in (AS.Generic, AS.Global, AS.Shared)
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


## PTX

# half-precision atomics using PTX instruction

for A in (AS.Generic, AS.Global, AS.Shared), T in (Float16,)
    if A == AS.Global
        scope = ".global"
    elseif A == AS.Shared
        scope = ".shared"
    else
        scope = ""
    end

    intr = "atom$scope.add.noftz.f16 \$0, [\$1], \$2;"
    @eval @inline atomic_add!(ptr::LLVMPtr{$T,$A}, val::$T) =
        @asmcall($intr, "=h,l,h", true, $T, Tuple{Core.LLVMPtr{$T,$A},$T}, ptr, val)
end


## Julia

# floating-point CAS via bitcasting

inttype(::Type{T}) where {T<:Integer} = T
inttype(::Type{Float16}) = Int16
inttype(::Type{Float32}) = Int32
inttype(::Type{Float64}) = Int64
inttype(::Type{BFloat16}) = Int16

for T in [Float16, Float32, Float64, BFloat16]
    @eval @inline function atomic_cas!(ptr::LLVMPtr{$T,A}, cmp::$T, new::$T) where {A}
        IT = inttype($T)
        cmp_i = reinterpret(IT, cmp)
        new_i = reinterpret(IT, new)
        old_i = atomic_cas!(reinterpret(LLVMPtr{IT,A}, ptr), cmp_i, new_i)
        return reinterpret($T, old_i)
    end
end


# generic atomic support using compare-and-swap

@inline function atomic_op!(ptr::LLVMPtr{T}, op::Function, val) where {T}
    old = Base.unsafe_load(ptr)
    while true
        cmp = old
        new = convert(T, op(old, val))
        old = atomic_cas!(ptr, cmp, new)
        isequal(old, cmp) && return new
    end
end


## documentation

"""
    atomic_cas!(ptr::LLVMPtr{T}, cmp::T, val::T)

Reads the value `old` located at address `ptr` and compare with `cmp`. If `old` equals to
`cmp`, stores `val` at the same address. Otherwise, doesn't change the value `old`. These
operations are performed in one atomic transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Additionally, on GPU hardware with compute capability 7.0+, values of type UInt16 are
supported.
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
"""
atomic_sub!

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
"""
atomic_min!

"""
    atomic_max!(ptr::LLVMPtr{T}, val::T)

Reads the value `old` located at address `ptr`, computes `max(old, val)`, and stores the
result back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
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

asm(::Type{LLVMOrdering{:monotonic}}) = :relaxed
asm(::Type{LLVMOrdering{Order}}) where Order = Order

asm(::Type{SystemScope}) = :sys
asm(::Type{DeviceScope}) = :gpu
asm(::Type{BlockScope}) = :cta

function suffix(sz)
    if sz == 1
        "b8"
    elseif sz == 2
        "b16"
    elseif sz == 4
        "b32"
    elseif sz == 8
        "b64"
    end
end

function reg(sz)
    if sz == 1
        "r"
    elseif sz == 2
        "h"
    elseif sz == 4
        "r"
    elseif sz == 8
        "l"
    end
end

function addr_space(A)
    if A == AS.Global
        as = ".global"
    elseif A == AS.Shared
        as = ".shared"
    else
        as = ""
    end
end

for (order, scope, A, sz) in Iterators.product(
                                (LLVMOrdering{:acquire}, LLVMOrdering{:monotonic}),
                                (BlockScope, DeviceScope, SystemScope),
                                (AS.Generic, AS.Global, AS.Shared),
                                (2,4,8))
    instruction = "ld.$(asm(order)).$(asm(scope))$(addr_space(A)).$(suffix(sz)) \$0, [\$1];"
    constraint  = "=$(reg(sz)),l,~{memory}"
    @eval @inline __load(::Val{$sz}, ptr::LLVMPtr{T, $A}, ::$order, ::$scope) where {T} =
        @asmcall($instruction, $constraint, true, T, Tuple{LLVMPtr{T, $A}}, ptr)
end

# Handle byte sized load
for (order, scope, A) in Iterators.product(
                            (LLVMOrdering{:acquire}, LLVMOrdering{:monotonic}),
                            (BlockScope, DeviceScope, SystemScope),
                            (AS.Generic, AS.Global, AS.Shared))
    instruction = "ld.$(asm(order)).$(asm(scope))$(addr_space(A)).b8 \$0, [\$1];"
    constraint  = "=r,l,~{memory}"
    @eval @inline function __load(::Val{1}, ptr::LLVMPtr{T, $A}, ::$order, ::$scope) where {T}
        val = @asmcall($instruction, $constraint, true, UInt32, Tuple{LLVMPtr{T, $A}}, ptr)
        return Core.bitcast(T, val % UInt8)
    end
end

@inline __load(ptr::LLVMPtr{T}, order, scope) where T =
    __load(Val(sizeof(T)), ptr, order, scope)

for (A, sz) in Iterators.product(
                    (AS.Generic, AS.Global, AS.Shared),
                    (2,4,8))
    instruction = "ld.volatile$(addr_space(A)).$(suffix(sz)) \$0, [\$1];"
    constraint  = "=$(reg(sz)),l,~{memory}"
    @eval @inline __load_volatile(::Val{$sz}, ptr::LLVMPtr{T, $A}) where {T} =
        @asmcall($instruction, $constraint, true, T, Tuple{LLVMPtr{T, $A}}, ptr)
end

# Handle byte sized load
for (A) in (AS.Generic, AS.Global, AS.Shared)
    instruction = "ld.volatile$(addr_space(A)).b8 \$0, [\$1];"
    constraint  = "=r,l,~{memory}"
    @eval @inline function __load_volatile(::Val{1}, ptr::LLVMPtr{T, $A}) where {T}
        val = @asmcall($instruction, $constraint, true, UInt32, Tuple{LLVMPtr{T, $A}}, ptr)
        return Core.bitcast(T, val % UInt8)
    end
end

@inline __load_volatile(ptr::LLVMPtr{T}) where {T} =
    __load_volatile(Val(sizeof(T)), ptr)

@inline function atomic_load(ptr::LLVMPtr{T}, order, scope::SyncScope=device_scope) where T
    if order == acq_rel || order == release
        throw(AtomicOrderUnsupported(order))
    end
    if compute_capability() >= sv"7.0"
        if order == monotonic
            val = __load(ptr, monotonic, scope)
            return val
        end
        if order == seq_cst
            atomic_thread_fence(seq_cst, scope)
        end
        val = __load(ptr, acquire, scope)
        return val
    elseif compute_capability() >= sv"6.0"
        if order == seq_cst
            atomic_thread_fence(seq_cst, scope)
        end
        val = __load_volatile(ptr)
        if order == monotonic
            return val
        end
        atomic_thread_fence(order, scope)
        return val
    else
        throw(AtomicUnsupported{T}())
    end
end

for (order, scope, A, sz) in Iterators.product(
                            (LLVMOrdering{:release}, LLVMOrdering{:monotonic}),
                            (BlockScope, DeviceScope, SystemScope),
                            (AS.Generic, AS.Global, AS.Shared),
                            (1, 2, 4, 8))
    instruction = "st$(addr_space(A)).$(asm(order)).$(asm(scope)).$(suffix(sz)) [\$0], \$1;"
    constraint  = "l,$(reg(sz)),~{memory}"
    @eval @inline __store!(::Val{$sz}, ptr::LLVMPtr{T, $A}, val::T, ::$order, ::$scope) where {T} =
        @asmcall($instruction, $constraint, true, Cvoid, Tuple{LLVMPtr{T, $A}, T}, ptr, val)
end

@inline __store!(ptr::LLVMPtr{T}, val::T, order, scope) where T =
    __store!(Val(sizeof(T)), ptr, val, order, scope)

for (A, sz) in Iterators.product(
                (AS.Generic, AS.Global, AS.Shared),
                (1, 2, 4, 8))
    instruction = "st$(addr_space(A)).volatile.$(suffix(sz)) [\$0], \$1;"
    constraint  = "l,$(reg(sz)),~{memory}"
    @eval @inline __store_volatile!(::Val{$sz}, ptr::LLVMPtr{T, $A}, val::T) where {T} =
        @asmcall($instruction, $constraint, true, Cvoid, Tuple{LLVMPtr{T, $A}, T}, ptr, val)
end

# Could be done using LLVM.
@inline __store_volatile!(ptr::LLVMPtr{T}, val::T) where {T} =
    __store_volatile(Val(sizeof(T)), ptr, val)

@inline function atomic_store!(ptr::LLVMPtr{T}, val::T, order, scope::SyncScope=device_scope) where T
    if order == acq_rel || order == acquire # || order == consume
        throw(AtomicOrderUnsupported(order))
    end
    if compute_capability() >= sv"7.0"
        if order == release
            __store!(ptr, val, release, scope)
            return
        end
        if order == seq_cst
            atomic_thread_fence(seq_cst, scope)
        end
        __store!(ptr, val, monotonic, scope)
    elseif compute_capability() >= sv"6.0"
        if order == seq_cst
            atomic_thread_fence(seq_cst, scope)
        end
        __store_volatile!(ptr, val)
    else
        throw(AtomicUnsupported{T}())
    end
end

order(::LLVMOrdering{:monotonic}) = 1
# order(::Consume) = 2
order(::LLVMOrdering{:acquire}) = 3
order(::LLVMOrdering{:release}) = 4
order(::LLVMOrdering{:acq_rel}) = 5
order(::LLVMOrdering{:seq_cst}) = 6

Base.isless(a::LLVMOrdering, b::LLVMOrdering) = isless(order(a), order(b))

@inline function stronger_order(a::LLVMOrdering, b::LLVMOrdering)
    m = max(a, b)
    if m != release
        return m
    end
    # maximum is release, what is the other one?
    other = min(a, b)
    if other == monotonic
        return release
    # elseif other == Consume()
    #     return Acq_Rel()
    elseif other == acquire
        return acq_rel
    elseif other == release
        return release
    end
    Base.llvmcall("unreachable", Cvoid, Tuple{})
    @assert(false)
end

for (order, scope, A, sz) in Iterators.product(
                                (LLVMOrdering{:acq_rel}, LLVMOrdering{:acquire}, LLVMOrdering{:monotonic}, LLVMOrdering{:release}),
                                (BlockScope, DeviceScope, SystemScope),
                                (AS.Generic, AS.Global, AS.Shared),
                                (4, 8))
    instruction = "atom$(addr_space(A)).cas.$(asm(order)).$(asm(scope)).$(suffix(sz)) \$0, [\$1], \$2, \$3;"
    constraint  = "=$(reg(sz)),l,$(reg(sz)),$(reg(sz)),~{memory}"
    @eval @inline __cas!(::Val{$sz}, ptr::LLVMPtr{T, $A}, old::T, new::T, ::$order, ::$scope) where {T} =
        @asmcall($instruction, $constraint, true, T, Tuple{LLVMPtr{T, $A}, T, T}, ptr, old, new)
end

@inline __cas!(ptr::LLVMPtr{T}, old::T, new::T, order, scope) where T =
    __cas!(sizeof(T), ptr, old, new, order, scope)

for (scope, A, sz) in Iterators.product(
                                (LLVMOrdering{:acq_rel}, LLVMOrdering{:acquire}, LLVMOrdering{:monotonic}, LLVMOrdering{:release}),
                                (AS.Generic, AS.Global, AS.Shared),
                                (2, 4, 8))
    instruction = "atom$(addr_space(A)).cas.$(asm(scope)).$(suffix(sz)) \$0, [\$1], \$2, \$3;"
    constraint  = "=$(reg(sz)),l,$(reg(sz)),$(reg(sz)),~{memory}"
    @eval @inline __cas!(::Val{$sz}, ptr::LLVMPtr{T, $A}, old::T, new::T, ::$scope) where {T} =
        @asmcall($instruction, $constraint, true, T, Tuple{LLVMPtr{T, $A}, T, T}, ptr, old, new)
end

@inline __cas!(ptr::LLVMPtr{T}, old::T, new::T, scope) where T =
    __cas!(Val(sizeof(T)), ptr, old, new, scope)

@inline function atomic_cas!(ptr::LLVMPtr{T}, expected::T, new::T, success_order, failure_order, scope::SyncScope=device_scope) where T
    order = stronger_order(success_order, failure_order)
    if compute_capability() >= sv"7.0" && 2 <= sizeof(T) <= 4
        if order == seq_cst
            atomic_thread_fence(seq_cst, scope)
        end
        if order == seq_cst # order == consume
            order = acquire
        end
        old = __cas!(ptr, expected, new, order, scope)
    elseif compute_capability() >= sv"6.0"
        if order == seq_cst || order == acq_rel || order == release
            atomic_thread_fence(seq_cst, scope)
        end
        old = __cas!(ptr, expected, new, scope)
        if order == seq_cst || order == acq_rel || order == acquire # order == consume
            atomic_thread_fence(seq_cst, scope)
        end
    else
        throw(AtomicUnsupported{T}())
    end
    return old
end

#
# High-level interface
#
import Atomix: @atomic, @atomicswap, @atomicreplace
# import UnsafeAtomicsLLVM

if VERSION <= v"1.7"
    export @atomic
end

using Atomix: Atomix, IndexableRef

const CuIndexableRef{Indexable<:CuDeviceArray} = IndexableRef{Indexable}

@inline function Atomix.get(ref::CuIndexableRef, order)
    atomic_load(Atomix.pointer(ref), order)
end

@inline function Atomix.set!(ref::CuIndexableRef, v, order)
    v = convert(eltype(ref), v)
    atomic_store!(Atomix.pointer(ref), v, order)
end

@inline function Atomix.replace!(ref::CuIndexableRef, expected, desired,
                                 success_ordering, failure_ordering)
    ptr = Atomix.pointer(ref)
    expected = convert(eltype(ref), expected)
    desired = convert(eltype(ref), desired)
    return atomic_cas!(ptr, expected, desired, success_ordering, failure_ordering)
end

@inline function modify!(ptr::LLVMPtr{T}, op::OP, x, order) where {T, OP}
    old = atomic_load(ptr, order)
    while true
        expected = old
        new = op(expected, x)
        old = atomic_cas!(ptr, expected, new, order, monotonic)
        if old === expected
            return expected => new
        end
    end
end

@inline function Atomix.modify!(ref::CuIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    # TODO: Support hardware variants
    # old = if op === (+)
    #     atomic_add!(ptr, x)
    # elseif op === (-)
    #     atomic_sub!(ptr, x)
    # elseif op === (&)
    #     atomic_and!(ptr, x)
    # elseif op === (|)
    #     atomic_or!(ptr, x)
    # elseif op === xor
    #     atomic_xor!(ptr, x)
    # elseif op === min
    #     atomic_min!(ptr, x)
    # elseif op === max
    #     atomic_max!(ptr, x)
    # else
        return modify!(ptr, op, x, order)
    # end
    # return old => op(old, x)
end
