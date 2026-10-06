# Atomic Functions (B.12)

#
# Low-level intrinsics
#

# These implement CUDA C's atomic functions on top of UnsafeAtomics, which emits LLVM
# atomics that the NVPTX back-end lowers (expanding what PTX lacks into compare-and-swap
# loops). Like CUDA C's, they are relaxed: use UnsafeAtomics directly for other orderings.
const atomic_order = UnsafeAtomics.monotonic

# Memory scopes, named after CUDA C's scoped variants. Default is device scope, like CUDA
# C's atomic*() (the _system/_block variants are explicit there too). UnsafeAtomics' own
# default is the system scope, which Pascal cannot provide under Windows (#3187).
const atomic_scopes = (:block, :device, :system)
const AtomicScope = Union{Val{:block}, Val{:device}, Val{:system}}

unsafe_atomics_scope(::Val{:block}) = UnsafeAtomics.workgroup
unsafe_atomics_scope(::Val{:device}) = UnsafeAtomics.device
unsafe_atomics_scope(::Val{:system}) = UnsafeAtomics.system

# Shared memory is confined to a block, regardless of the requested scope.
@inline function check_atomic_scope(::LLVMPtr{T,A}, ::Val{S}) where {T,A,S}
    if S === :system && A != AS.Shared
        GPUCompiler.@static_assert(compute_capability() >= sv"6.0",
            "system-scope atomics require compute capability 6.0; use device scope")
        @static if Sys.iswindows()
            GPUCompiler.@static_assert(compute_capability() >= sv"7.0",
                "system-scope atomics are not supported on Pascal GPUs under Windows; use device scope")
        end
    end
    return
end

# PTX only has atomics on global and shared memory, which generic pointers may point to.
const AtomicPtr{T} = Union{LLVMPtr{T,AS.Generic}, LLVMPtr{T,AS.Global}, LLVMPtr{T,AS.Shared}}

@inline function atomic_rmw!(f::F, ptr::LLVMPtr, val, scope::AtomicScope) where {F}
    check_atomic_scope(ptr, scope)
    f(ptr, val, atomic_order, unsafe_atomics_scope(scope))
end

for (fn, rmw, types) in [(:atomic_add!,  UnsafeAtomics.add!,
                          (Int32, Int64, UInt32, UInt64, Float16, Float32, Float64)),
                         (:atomic_sub!,  UnsafeAtomics.sub!,  (Int32, Int64, UInt32, UInt64)),
                         (:atomic_and!,  UnsafeAtomics.and!,  (Int32, Int64, UInt32, UInt64)),
                         (:atomic_or!,   UnsafeAtomics.or!,   (Int32, Int64, UInt32, UInt64)),
                         (:atomic_xor!,  UnsafeAtomics.xor!,  (Int32, Int64, UInt32, UInt64)),
                         (:atomic_min!,  UnsafeAtomics.min!,  (Int32, Int64, UInt32, UInt64)),
                         (:atomic_max!,  UnsafeAtomics.max!,  (Int32, Int64, UInt32, UInt64)),
                         (:atomic_xchg!, UnsafeAtomics.xchg!, (Int32, Int64, UInt32, UInt64))]
    for T in types
        @eval @inline $fn(ptr::AtomicPtr{$T}, val::$T, scope::AtomicScope=Val(:device)) =
            atomic_rmw!($rmw, ptr, val, scope)
    end
end

# PTX only has BFloat16 addition from sm_90, and LLVM only knows BFloat16s.BFloat16 as such
# when it is Core.BFloat16. otherwise, loop on `atomic_cas!` (see `atomic_cas_b16`).
@inline function atomic_add!(ptr::AtomicPtr{BFloat16}, val::BFloat16,
                             scope::AtomicScope=Val(:device))
    @static if isdefined(Core, :BFloat16) && BFloat16 === Core.BFloat16
        compute_capability() >= sv"9.0" && return atomic_rmw!(UnsafeAtomics.add!, ptr, val, scope)
    end
    first(atomic_modify!(ptr, +, val, scope))
end

# PTX has no floating-point subtraction, so add the negated value instead of having the
# back-end expand `atomicrmw fsub` into a compare-and-swap loop.
for T in (Float16, Float32, Float64, BFloat16)
    @eval @inline atomic_sub!(ptr::AtomicPtr{$T}, val::$T, scope::AtomicScope=Val(:device)) =
        atomic_add!(ptr, -val, scope)
end

for T in (Int16, UInt16, Int32, Int64, UInt32, UInt64, Float16, Float32, Float64, BFloat16)
    @eval @device_function @inline function atomic_cas!(ptr::LLVMPtr{$T,A}, cmp::$T, val::$T,
                                                        scope::AtomicScope=Val(:device)) where {A}
        GPUCompiler.@static_assert(
            A == AS.Generic || A == AS.Global || A == AS.Shared,
            "atomics require a generic, global, or shared address space")
        check_atomic_scope(ptr, scope)
        @static if sizeof($T) == 2
            compute_capability() >= sv"7.0" && return atomic_cas_b16(ptr, cmp, val, scope)
        end
        UnsafeAtomics.cas!(ptr, cmp, val, atomic_order, atomic_order,
                           unsafe_atomics_scope(scope)).old
    end
end

# The hardware has no 16-bit compare-and-swap: both LLVM (llvm/llvm-project#120220) and
# ptxas (for PTX's `atom.cas.b16`) emulate it with a loop on the containing 32-bit word.
# ptxas does it better, as it is not bound by the PTX memory model: it reads the initial
# word with a weak load and doesn't yield in the loop, while LLVM has to use a relaxed load
# (llvm/llvm-project#188361) and ptxas inserts a YIELD in every iteration of LLVM's loop.
# That makes compare-and-swap loops up to 30% slower under contention (RTX 5080), so use
# the native instruction where PTX has it, like CUDA C does. It also keeps compute-sanitizer
# from flagging accesses to the neighbouring value in memory that isn't padded to whole
# 32-bit words.
ptx_scope(::Val{:block}) = ".cta"
ptx_scope(::Val{:device}) = ".gpu"
ptx_scope(::Val{:system}) = ".sys"
for A in (AS.Generic, AS.Global, AS.Shared), S in atomic_scopes
    space = A == AS.Global ? ".global" : A == AS.Shared ? ".shared" : ""
    intr = "atom.relaxed$(ptx_scope(Val(S)))$space.cas.b16 \$0, [\$1], \$2, \$3;"
    @eval @device_function @inline atomic_cas_b16(ptr::LLVMPtr{UInt16,$A}, cmp::UInt16,
                                                  val::UInt16, ::Val{$(QuoteNode(S))}) =
        @asmcall($intr, "=h,l,h,h", true, UInt16,
                 Tuple{LLVMPtr{UInt16,$A},UInt16,UInt16}, ptr, cmp, val)
end
@inline atomic_cas_b16(ptr::LLVMPtr{T,A}, cmp::T, val::T, scope::Val) where {T,A} =
    reinterpret(T, atomic_cas_b16(reinterpret(LLVMPtr{UInt16,A}, ptr), reinterpret(UInt16, cmp),
                                  reinterpret(UInt16, val), scope))

# CUDA C's atomicInc and atomicDec, which interpret the value as unsigned, are LLVM's
# `uinc_wrap` and `udec_wrap`. LLVM 15 can't express those: there, the back-end upgrades
# NVVM's device-scope intrinsics, and UnsafeAtomics uses a compare-and-swap loop otherwise.
for A in (AS.Generic, AS.Global, AS.Shared), (fn, rmw, op) in
        [(:atomic_inc!, UnsafeAtomics.inc_wrap!, :inc), (:atomic_dec!, UnsafeAtomics.dec_wrap!, :dec)]
    @static if Base.libllvm_version < v"16"
        intr = "llvm.nvvm.atomic.load.$op.32.p$(convert(Int, A))i32"
        @eval @device_function @inline $fn(ptr::LLVMPtr{Int32,$A}, val::Int32, ::Val{:device}) =
            @typed_ccall($intr, llvmcall, Int32, (LLVMPtr{Int32,$A}, Int32), ptr, val)
    end
    @eval @inline function $fn(ptr::LLVMPtr{Int32,$A}, val::Int32,
                               scope::AtomicScope=Val(:device))
        old = atomic_rmw!($rmw, reinterpret(LLVMPtr{UInt32,$A}, ptr),
                          reinterpret(UInt32, val), scope)
        reinterpret(Int32, old)
    end
end


## generic atomic support using compare-and-swap

# Returns `(old, new)`. The result of `op` is converted to the element type, so `op` can
# promote (e.g. `/` on integers).
@inline function atomic_modify!(ptr::LLVMPtr{T}, op::Function, val,
                                scope::AtomicScope=Val(:device)) where {T}
    check_atomic_scope(ptr, scope)
    sizeof(T) == 2 && return atomic_modify_b16!(ptr, op, val, scope)
    old, new = UnsafeAtomics.modify!(ptr, (old, val) -> convert(T, op(old, val)),
                                     convert(T, val), atomic_order,
                                     unsafe_atomics_scope(scope))
    return old, new
end

# loop on `atomic_cas!`, which uses the native 16-bit instruction where available (see
# `atomic_cas_b16`), instead of on LLVM's emulation of it like `UnsafeAtomics.modify!` does
@inline function atomic_modify_b16!(ptr::LLVMPtr{T}, op::Function, val,
                                    scope::AtomicScope) where {T}
    old = UnsafeAtomics.load(ptr, atomic_order, unsafe_atomics_scope(scope))
    while true
        new = convert(T, op(old, val))
        cur = atomic_cas!(ptr, old, new, scope)
        # compare bits, so that a NaN with another payload doesn't count as success
        reinterpret(UInt16, cur) == reinterpret(UInt16, old) && return old, new
        old = cur
    end
end

@inline atomic_op!(ptr::LLVMPtr, op::Function, val, scope::AtomicScope=Val(:device)) =
    last(atomic_modify!(ptr, op, val, scope))


## documentation

"""
    atomic_cas!(ptr::LLVMPtr{T}, cmp::T, val::T, [scope::Val])

Reads the value `old` located at address `ptr` and compare with `cmp`. If `old` equals to
`cmp`, stores `val` at the same address. Otherwise, doesn't change the value `old`. These
operations are performed in one atomic transaction. The function returns `old`.

This operation is supported for values of type Int16, Int32, Int64, UInt16, UInt32,
UInt64, Float16, Float32, Float64, and BFloat16. 16-bit operations are implemented with a
compare-and-swap of the 32-bit word that contains the value before compute capability 7.0.

Like CUDA C's atomic functions, these operations are relaxed: they don't order the memory
accesses around them. For other orderings, use UnsafeAtomics.jl, or Atomix.jl's `@atomic`.

`scope` selects the set of threads the operation is atomic with respect to: `Val(:block)`,
`Val(:device)` (default, matching CUDA C's `atomicX`), or `Val(:system)` (matching
`atomicX_system`; requires compute capability 6.0, or 7.2 on Tegra, and is not available on
Pascal GPUs under Windows).
"""
atomic_cas!

"""
    atomic_xchg!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr` and stores `val` at the same address. These
operations are performed in one atomic transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_xchg!

"""
    atomic_add!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `old + val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32, UInt64, Float16,
Float32, Float64, and BFloat16. The back-end uses a native instruction where available and
emulates the operation otherwise.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_add!

"""
    atomic_sub!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `old - val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32, UInt64, Float16,
Float32, Float64, and BFloat16.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_sub!

"""
    atomic_and!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `old & val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_and!

"""
    atomic_or!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `old | val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_or!

"""
    atomic_xor!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `old ⊻ val`, and stores the result
back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_xor!

"""
    atomic_min!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `min(old, val)`, and stores the
result back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_min!

"""
    atomic_max!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `max(old, val)`, and stores the
result back to memory at the same address. These operations are performed in one atomic
transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_max!

"""
    atomic_inc!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `((old >= val) ? 0 : (old+1))`, and
stores the result back to memory at the same address. These three operations are performed
in one atomic transaction. The function returns `old`.

This operation accepts Int32 values, interpreted as unsigned 32-bit integers.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
"""
atomic_inc!

"""
    atomic_dec!(ptr::LLVMPtr{T}, val::T, [scope::Val])

Reads the value `old` located at address `ptr`, computes `(((old == 0) | (old > val)) ? val
: (old-1) )`, and stores the result back to memory at the same address. These three
operations are performed in one atomic transaction. The function returns `old`.

This operation accepts Int32 values, interpreted as unsigned 32-bit integers.

For memory scopes, ordering and platform restrictions, see [`atomic_cas!`](@ref).
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

Like the lower-level `atomic_...!` functions, these operations are relaxed and atomic with
respect to the threads of the device.

!!! warn
    This interface is experimental, and might change without warning. Prefer Atomix.jl's
    `@atomic` (as used by KernelAbstractions.jl), which also supports other orderings. Note
    that, like `Base.@atomic`, it reads an assignment `@atomic a[I] = a[I] + val` as an
    atomic store of a value that is computed separately; write `@atomic a[I] += val` for an
    atomic update.
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
@public @atomic, AtomicError, atomic_add!, atomic_sub!, atomic_and!, atomic_or!, atomic_xor!, atomic_min!, atomic_max!, atomic_inc!, atomic_dec!, atomic_cas!, atomic_xchg!

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

# native atomics that the back-end expands on older devices
@inline atomic_arrayset(A::AbstractArray{T}, I::Integer, ::typeof(+), val::T) where
        {T <: Union{Float16,Float64,BFloat16}} = atomic_add!(pointer(A, I), val)

# fallback using compare-and-swap
@inline atomic_arrayset(A::AbstractArray{T}, I::Integer, op::Function, val) where {T} =
    atomic_op!(pointer(A, I), op, val)
