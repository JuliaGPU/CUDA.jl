# Atomic Functions (B.12)

#
# Low-level intrinsics
#

# TODO: We should use LLVM intrinsics wherever possible.

abstract type ThreadScope end
struct System <: ThreadScope end
struct Device <: ThreadScope end
struct Block <: ThreadScope end

asm(::Type{System}) = "sys"
asm(::Type{Device}) = "gpu"
asm(::Type{Block}) = "cta"

abstract type MemoryOrder end
struct Relaxed <: MemoryOrder end
struct Consume <: MemoryOrder end
struct Acquire <: MemoryOrder end
struct Release <: MemoryOrder end
struct Acq_Rel <: MemoryOrder end
struct Seq_Cst <: MemoryOrder end

asm(::Type{Relaxed}) = "relaxed"
asm(::Type{Consume}) = "consume"
asm(::Type{Acquire}) = "acquire"
asm(::Type{Release}) = "release"
asm(::Type{Acq_Rel}) = "acq_rel"
asm(::Type{Seq_Cst}) = "seq_cst"

order(::Relaxed) = 1
order(::Consume) = 2
order(::Acquire) = 3
order(::Release) = 4
order(::Acq_Rel) = 5
order(::Seq_Cst) = 6

Base.isless(a::MemoryOrder, b::MemoryOrder) = isless(order(a), order(b))

function stronger_order(a::MemoryOrder, b::MemoryOrder)
    m = max(a, b)
    if m != Release()
        return m
    end
    # maximum is release, what is the other one?
    other = min(a, b)
    if other == Relaxed()
        return Release()
    elseif other == Consume()
        return Acq_Rel()
    elseif other == Acquire()
        return Acq_Rel()
    elseif other == Release()
        return Release()
    end
    assert(false)
end

threadfence_sc_block() =
    @asmcall("fence.sc.cta;", "~{memory}", true, Cvoid, Tuple{})
threadfence_acq_rel_block() =
    @asmcall("fence.acq_rel.cta;", "~{memory}", true, Cvoid, Tuple{})

function atomic_thread_fence(order, scope::Block)
    if compute_capability() >= sv"7.0"
        if order == Seq_Cst()

            threadfence_sc_block()
        elseif order == Consume() ||
               order == Acquire() ||
               order == Acq_Rel() ||
               order == Release()

            threadfence_acq_rel_block()
        else 
            assert(false)
        end
    else
        if order == Seq_Cst() ||
           order == Consume() ||
           order == Acquire() ||
           order == Acq_Rel() ||
           order == Release()

            threadfence_block()
        else
            assert(false)
        end
    end
end

threadfence_sc_system() =
    @asmcall("fence.sc.sys;", "~{memory}", true, Cvoid, Tuple{})
threadfence_acq_rel_system() =
    @asmcall("fence.acq_rel.sys;", "~{memory}", true, Cvoid, Tuple{})

function atomic_thread_fence(order, scope::Device)
    if compute_capability() >= sv"7.0"
        if order == Seq_Cst()

            threadfence_sc_device()
        elseif order == Consume() ||
               order == Acquire() ||
               order == Acq_Rel() ||
               order == Release()

            threadfence_acq_rel_device()
        else 
            assert(false)
        end
    else
        if order == Seq_Cst() ||
           order == Consume() ||
           order == Acquire() ||
           order == Acq_Rel() ||
           order == Release()

            threadfence()
        else
            assert(false)
        end
    end
end

function atomic_thread_fence(order, scope::System=System())
    if compute_capability() >= sv"7.0"
        if order == Seq_Cst()

            threadfence_sc_system()
        elseif order == Consume() ||
               order == Acquire() ||
               order == Acq_Rel() ||
               order == Release()

            threadfence_acq_rel_system()
        else 
            assert(false)
        end
    else
        if order == Seq_Cst() ||
           order == Consume() ||
           order == Acquire() ||
           order == Acq_Rel() ||
           order == Release()

            threadfence_system()
        else
            assert(false)
        end
    end
end

for (order, scope) in Iterators.product((Acquire, Relaxed), 
                                        (Block, Device, System))
    asm_b64 = "ld.$(asm(order)).$(asm(scope)).b64 %0, [%1];"
    asm_b32 = "ld.$(asm(order)).$(asm(scope)).b32 %0, [%1];"
    @eval __load_64(ptr::LLVMPtr{T}, ::$order, ::$scope) where T =  @asmcall($asm_b64, "=l,l,~{memory}", true, T, Tuple{LLVMPtr{T}}, ptr)
    @eval __load_32(ptr::LLVMPtr{T}, ::$order, ::$scope) where T =  @asmcall($asm_b32, "=r,l,~{memory}", true, T, Tuple{LLVMPtr{T}}, ptr)
end

function __load(ptr::LLVMPtr{T}, order, scope) where T
    if sizeof(T) == 4
        __load_32(ptr, order, scope)
    elseif sizeof(T) == 8
        __load_64(ptr, order, scope)
    else
        assert(false)
    end
end

# Could be done using LLVM.
__load_volatile_64(ptr::LLVMPtr{T}) where T = @asmcall("ld.volatile.b64 %0, [%1];", "=l,l,~{memory}", true, T, Tuple{LLVMPtr{T}})
__load_volatile_32(ptr::LLVMPtr{T}) where T = @asmcall("ld.volatile.b32 %0, [%1];", "=r,l,~{memory}", true, T, Tuple{LLVMPtr{T}})

function __load_volatile(ptr::LLVMPtr{T}) where T
    if sizeof(T) == 4
        __load_volatile_32(ptr, order, scope)
    elseif sizeof(T) == 8
        __load_volatile_64(ptr, order, scope)
    else
        assert(false)
    end
end

function atomic_load(ptr::LLVMPtr{T}, order, scope::System=System()) where T
    if order == Acq_Rel() || order == Release()
        assert(false)
    end
    if compute_capability() >= sv"7.0"
        if order == Relaxed()
            val = __load(ptr, Relaxed(), scope)
            return val
        end
        if order == Seq_Cst()
            atomic_thread_fence(Seq_Cst(), scope)
        end
        val = __load(ptr, Acquire(), scope)
        return val
    else
        if order == Seq_Cst()
            atomic_thread_fence(Seq_Cst(), scope)
        end
        val = __load_volatile(ptr)
        if order == Relaxed()
            return val
        end
        atomic_thread_fence(order, scope)
        return val
    end
end

for (order, scope) in Iterators.product((Acquire, Relaxed), 
                                        (Block, Device, System))
    asm_b64 = "st.$(asm(order)).$(asm(scope)).b64 [%0], %1;"
    asm_b32 = "st.$(asm(order)).$(asm(scope)).b32 [%0], %1;"
    @eval __store_64!(ptr::LLVMPtr{T}, val::T, ::$order, ::$scope) where T =  @asmcall($asm_b64, "l,l,~{memory}", true, Cvoid, Tuple{LLVMPtr{T}, T}, ptr, val)
    @eval __store_32!(ptr::LLVMPtr{T}, val::T, ::$order, ::$scope) where T =  @asmcall($asm_b32, "l,r,~{memory}", true, Cvoid, Tuple{LLVMPtr{T}, T}, ptr, val)
end

function __store!(ptr::LLVMPtr{T}, val::T, order, scope) where T
    if sizeof(T) == 4
        __store_32!(ptr, val, order, scope)
    elseif sizeof(T) == 8
        __store_64!(ptr, val, order, scope)
    else
        assert(false)
    end
end

# Could be done using LLVM.
__store_volatile_32!(ptr::LLVMPtr{T}, val::T)  where T = @asmcall("st.volatile.b32 [%0], %1;", "l,r,~{memory}", true, Cvoid, Tuple{LLVMPtr{T}, T}, ptr, val)
__store_volatile_64!(ptr::LLVMPtr{T}, val::T)  where T = @asmcall("st.volatile.b64 [%0], %1;", "l,l,~{memory}", true, Cvoid, Tuple{LLVMPtr{T}, T}, ptr, val)

function __store_volatile!(ptr::LLVMPtr{T}, val::T) where T
    if sizeof(T) == 4
        __store_volatile_32!(ptr, val)
    elseif sizeof(T) == 8
        __store_volatile_64!(ptr, val)
    else
        assert(false)
    end
end

function atomic_store!(ptr::LLVMPtr{T}, val::T, order, scope::System=System()) where T
    if order == Acq_Rel() || order == Consume() || order == Acquire()
        assert(false)
    end
    if compute_capability() >= sv"7.0"
        if order == Release()
            __store!(ptr, val, Release(), scope)
            return
        end
        if order == Seq_Cst()
            atomic_thread_fence(Seq_Cst(), scope)
        end
        __store!(ptr, val, Relaxed(), scope)
    else
        if order == Seq_Cst()
            atomic_thread_fence(Seq_Cst(), scope)
        end
        __store_volatile!(ptr, val)
    end
end

for (order, scope) in Iterators.product((Acq_Rel, Acquire, Relaxed, Release), 
                                        (Block, Device, System))
    asm_b64 = "atom.cas.$(asm(order)).$(asm(scope)).b64 %0,[%1],%2,%3;"
    asm_b32 = "atom.cas.$(asm(order)).$(asm(scope)).b32 %0,[%1],%2,%3;"
    @eval __cas_64!(ptr::LLVMPtr{T}, old::T, new::T, ::$order, ::$scope) where T =  @asmcall($asm_b64, "=l,l,l,l,~{memory}", true, T, Tuple{LLVMPtr{T}, T, T}, ptr, old, new)
    @eval __cas_32!(ptr::LLVMPtr{T}, old::T, new::T, ::$order, ::$scope) where T =  @asmcall($asm_b32, "=r,l,r,r,~{memory}", true, T, Tuple{LLVMPtr{T}, T, T}, ptr, old, new)
end

function __cas!(ptr::LLVMPtr{T}, old::T, new::T, order, scope) where T
    if sizeof(T) == 4
        __cas_32!(ptr, old, new, order, scope)
    elseif sizeof(T) == 8
        __cas_64!(ptr, old, new, order, scope)
    else
        assert(false)
    end
end

for scope in (Block, Device, System)
    asm_b64 = "atom.cas.$(asm(scope)).b64 %0,[%1],%2,%3;"
    asm_b32 = "atom.cas.$(asm(scope)).b32 %0,[%1],%2,%3;"
    @eval __cas_volatile_64!(ptr::LLVMPtr{T}, old::T, new::T, ::$scope) where T =  @asmcall($asm_b64, "=l,l,l,l,~{memory}", true, T, Tuple{LLVMPtr{T}, T, T}, ptr, old, new)
    @eval __cas_volatile_32!(ptr::LLVMPtr{T}, old::T, new::T, ::$scope) where T =  @asmcall($asm_b32, "=r,l,r,r,~{memory}", true, T, Tuple{LLVMPtr{T}, T, T}, ptr, old, new)
end

function __cas_volatile!(ptr::LLVMPtr{T}, old::T, new::T, scope) where T
    if sizeof(T) == 4
        __cas__volatile_32!(ptr, old, new, scope)
    elseif sizeof(T) == 8
        __cas__volatile_64!(ptr, old, new, scope)
    else
        assert(false)
    end
end

function atomic_cas!(ptr::LLVMPtr{T}, old::T, new::T, success_order, failure_order, scope::System=System()) where T
    order = stronger_order(success_order, failure_order)
    if compute_capability() >= sv"7.0"
        if order == Seq_Cst()
            atomic_thread_fence(Seq_Cst(), scope)
        end
        if order == Consume() || order == Seq_Cst()
            order = Acquire()
        end
        val = __cas!(ptr, old, new, order, scope)
    else
        if order == Seq_Cst() || order == Acq_Rel() || order == Release()
            atomic_thread_fence(Seq_Cst(), scope)
        end
        val = __cas_volatile!(ptr, old, new, scope)
        if order == Seq_Cst() || order == Acq_Rel() || order == Consume() || order == Acquire()
            atomic_thread_fence(Seq_Cst(), scope)
        end
    end
    success = val == old
    return success => val
end

const OPS = ("exch", "add", "and", "or", "xor", "max", "min")
const OPS_SUFFIX = Dict(
    "exch" => "b",
    "add" => "u",
    "and" => "b",
    "or"  => "b",
    "xor" => "b",
    "max" => "u",
    "min" => "u"
)

for (bits, op) in Iterators.product((64, 32), OPS)
    op_sym = Symbol("__$(op)_$(bits)!")
    op_volatile_sym = Symbol("__$(op)_volatile_$(bits)!")
    ub = OPS_SUFFIX[op]
    if bits == 32
        constraints = "=r,l,r,~{memory}"
    else
        constraints = "=l,l,l,~{memory}"
    end
    for (order, scope) in Iterators.product((Acquire, Relaxed, Release, Acq_Rel), 
                                            (Block, Device, System))

        asm_s = "atom.$op.$(asm(order)).$(asm(scope)).$ub$bits %0,[%1],%2;"
        @eval $op_sym(ptr::LLVMPtr{T}, val::T, ::$order, ::$scope) where T = @asmcall($asm_s, $constraints, true, T, Tuple{LLVMPtr{T}, T}, ptr, val)
    end

    for scope in (Block, Device, System)
        asm_s = "atom.$op.$(asm(scope)).$ub$bits  %0,[%1],%2;"
        @eval $op_volatile_sym(ptr::LLVMPtr{T}, val::T, ::$scope) where T = @asmcall($asm_s, $constraints, true, T, Tuple{LLVMPtr{T}, T}, ptr, val)
    end
end

for op in OPS
    op_sym = Symbol("__$(op)!")
    op32_sym = Symbol("__$(op)_32!")
    op64_sym = Symbol("__$(op)_64!")
    op_volatile_sym = Symbol("__$(op)_volatile!")
    op32_volatile_sym = Symbol("__$(op)_volatile_32!")
    op64_volatile_sym = Symbol("__$(op)_volatile_64!")

    @eval function $op_sym(ptr::LLVMPtr{T}, val::T, order, scope) where T
        if sizeof(T) == 4
            $op32_sym(ptr, val, order, scope)
        elseif sizeof(T) == 8
            $op64_sym(ptr, val, order, scope)
        else
            assert(false)
        end
    end

    @eval function $op_volatile_sym(ptr::LLVMPtr{T}, val::T, scope) where T
        if sizeof(T) == 4
            $op32_volatile_sym(ptr, val, scope)
        elseif sizeof(T) == 8
            $op64_volatile_sym(ptr, val, scope)
        else
            assert(false)
        end
    end

    atomic_op_sym = Symbol("atomic_$(op)!")

    @eval function $atomic_op_sym(ptr::LLVMPtr{T}, val::T, order, scope::System=System()) where T
        if compute_capability() >= sv"7.0"
            if order == Seq_Cst()
                atomic_thread_fence(Seq_Cst(), scope)
            end
            if order == Seq_Cst() || order == Consume()
                order = Acquire()
            end
            val = $op_sym(ptr, val, order, scope)
        else
            if order == Seq_Cst() || order == Acq_Rel() || order == Release()
                atomic_thread_fence(Seq_Cst(), scope)
            end
            val = $op_volatile_sym(ptr, val, scope)
            if order == Seq_Cst() || order === Acq_Rel() || order == Consume() || order == Acquire()
                atomic_thread_fence(Seq_Cst(), scope)
            end
        end
        return val
    end
end

function atomic_sub!(ptr::LLVMPtr{T}, val::T, order, scope::System=System()) where T
    atomic_add!(ptr, -val, order, scope)
end

# TODO: "Derived" implementations for `sizeof(T) <= 2`

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
        T_val = convert(LLVMType, T)
        T_ptr = convert(LLVMType, ptr)

        T_typed_ptr = LLVM.PointerType(T_val, A)

        llvm_f, _ = create_function(T_val, [T_ptr, T_val])

        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
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

for T in (:Float32, :Float64)
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
        T_val = convert(LLVMType, T)
        T_ptr = convert(LLVMType, ptr)

        T_typed_ptr = LLVM.PointerType(T_val, A)

        llvm_f, _ = create_function(T_val, [T_ptr, T_val, T_val])

        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
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

for T in (:Int32, :Int64, :UInt32, :UInt64)
    @eval @inline atomic_cas!(ptr::LLVMPtr{$T}, cmp::$T, val::$T) =
        llvm_atomic_cas(ptr, cmp, val)
end

# NVPTX doesn't support cmpxchg with i16 yet
for A in (AS.Generic, AS.Global, AS.Shared), T in (:Int16, :UInt16)
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

for A in (AS.Generic, AS.Global, AS.Shared), T in (:Float16,)
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

for T in [:Float16, :Float32, :Float64, :BFloat16]
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



#
# High-level interface
#

# prototype of a high-level interface for performing atomic operations on arrays
#
# this design could be generalized by having atomic {field,array}{set,ref} accessors, as
# well as acquire/release operations to implement the fallback functionality where any
# operation can be applied atomically.

const inplace_ops = Dict(
    :(+=)   => :(+),
    :(-=)   => :(-),
    :(*=)   => :(*),
    :(/=)   => :(/),
    :(\=)   => :(\),
    :(%=)   => :(%),
    :(^=)   => :(^),
    :(&=)   => :(&),
    :(|=)   => :(|),
    :(⊻=)   => :(⊻),
    :(>>>=) => :(>>>),
    :(>>=)  => :(>>),
    :(<<=)  => :(<<),
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
    `atomic_...!` functions for a stable API, albeit one limited to natively-supported ops.
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
@inline atomic_arrayset(A::AbstractArray{T}, Is::Tuple, op::Function, val) where {T} =
    atomic_arrayset(A, Base._to_linear_index(A, Is...), op, convert(T, val))

# native atomics
for (op,impl,typ) in [(:(+), :(atomic_add!), [:UInt32,:Int32,:UInt64,:Int64,:Float32]),
                      (:(-), :(atomic_sub!), [:UInt32,:Int32,:UInt64,:Int64,:Float32]),
                      (:(&), :(atomic_and!), [:UInt32,:Int32,:UInt64,:Int64]),
                      (:(|), :(atomic_or!),  [:UInt32,:Int32,:UInt64,:Int64]),
                      (:(⊻), :(atomic_xor!), [:UInt32,:Int32,:UInt64,:Int64]),
                      (:max, :(atomic_max!), [:UInt32,:Int32,:UInt64,:Int64]),
                      (:min, :(atomic_min!), [:UInt32,:Int32,:UInt64,:Int64])]
    @eval @inline atomic_arrayset(A::AbstractArray{T}, I::Integer, ::typeof($op),
                                  val::T) where {T<:Union{$(typ...)}} =
        $impl(pointer(A, I), val)
end

# native atomics that are not supported on all devices
@inline function atomic_arrayset(A::AbstractArray{T}, I::Integer, op::typeof(+),
                                 val::T) where {T <: Union{Float64}}
    ptr = pointer(A, I)
    if compute_capability() >= sv"6.0"
        atomic_add!(ptr, val)
    else
        atomic_op!(ptr, op, val)
    end
end

# fallback using compare-and-swap
@inline atomic_arrayset(A::AbstractArray{T}, I::Integer, op::Function, val) where {T} =
    atomic_op!(pointer(A, I), op, val)
