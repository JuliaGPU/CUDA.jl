# Julia functions as CCCL operators
#
# HostJIT links operators supplied as LLVM IR into the module with the CUB kernels before
# generating PTX, so they inline. It calls them through a C ABI: every operand, and the
# result, is passed by (generic) pointer:
#
#     void op(void *a, void *b, void *out);               // stateless
#     void op(void *state, void *a, void *b, void *out);  // stateful
#
# where `state` points to a copy of the bytes the operator was called with. We compile a
# small Julia wrapper with that signature using GPUCompiler, so the operator itself can be
# any Julia function, including closures capturing bits values (passed as state).

const OpPtr = LLVMPtr{UInt8,0}

@inline load(::Type{T}, ptr::OpPtr) where {T} =
    unsafe_load(reinterpret(LLVMPtr{T,0}, ptr), 1, Val(Base.datatype_alignment(T)))
@inline store!(ptr::OpPtr, val::T) where {T} =
    unsafe_store!(reinterpret(LLVMPtr{T,0}, ptr), val, 1, Val(Base.datatype_alignment(T)))

# operators without state: the function is a singleton, recreated in the wrapper
struct StatelessOp{F,A,B,R} end
@inline function (::StatelessOp{F,A,B,R})(a::OpPtr, b::OpPtr, out::OpPtr) where {F,A,B,R}
    f = F.instance
    store!(out, convert(R, f(load(A, a), load(B, b))))
    return
end

# operators with state: the function object itself is the state
struct StatefulOp{F,A,B,R} end
@inline function (::StatefulOp{F,A,B,R})(state::OpPtr, a::OpPtr, b::OpPtr,
                                         out::OpPtr) where {F,A,B,R}
    f = load(F, state)
    store!(out, convert(R, f(load(A, a), load(B, b))))
    return
end

# unary operators (e.g. predicates) drop the second operand
struct StatelessUnaryOp{F,A,R} end
@inline function (::StatelessUnaryOp{F,A,R})(a::OpPtr, out::OpPtr) where {F,A,R}
    f = F.instance
    store!(out, convert(R, f(load(A, a))))
    return
end

struct StatefulUnaryOp{F,A,R} end
@inline function (::StatefulUnaryOp{F,A,R})(state::OpPtr, a::OpPtr,
                                           out::OpPtr) where {F,A,R}
    f = load(F, state)
    store!(out, convert(R, f(load(A, a))))
    return
end

isstateless(F::Type) = Base.issingletontype(F)

# a compiled operator, owning the memory its `cccl_op_t` refers to
mutable struct Operator
    const name::String
    const code::Vector{UInt8}
    const stateful::Bool
    const size::Int
    const alignment::Int
end

# operators are compiled once per wrapper type and device
const operators = Dict{Tuple{Type,CuDevice},Operator}()
const operators_lock = ReentrantLock()  # guards `operators`
const compile_lock = ReentrantLock()    # held while compiling
const operator_counter = Threads.Atomic{Int}(0)

"""
    CCCL.operator(f, A, B, R) -> Operator
    CCCL.operator(f, A, R) -> Operator

Compile the binary function `f`, called with arguments of type `A` and `B` (or the unary
function `f`, called with an argument of type `A`) and returning a value convertible to `R`,
to an operator for use with CCCL algorithms.
"""
function operator(f::F, A::Type, B::Type, R::Type) where {F}
    W = isstateless(F) ? StatelessOp{F,A,B,R} : StatefulOp{F,A,B,R}
    cached_operator(f, W, 3)
end

function operator(f::F, A::Type, R::Type) where {F}
    W = isstateless(F) ? StatelessUnaryOp{F,A,R} : StatefulUnaryOp{F,A,R}
    cached_operator(f, W, 2)
end

# `nargs` is the number of pointers a stateless wrapper `W` takes (operands and result)
function cached_operator(f::F, W::Type, nargs::Int) where {F}
    isbitstype(F) ||
        throw(ArgumentError("CCCL operators must be bits values; $F captures non-bits state"))
    stateless = isstateless(F)
    dev = device()
    lookup() = Base.@lock operators_lock get(operators, (W, dev), nothing)
    op = lookup()
    op === nothing || return op

    # compile without blocking lookups of other operators
    Base.@lock compile_lock begin
        op = lookup()
        op === nothing || return op

        name = "cccl_julia_op_$(Threads.atomic_add!(operator_counter, 1))"
        code = compile_operator(W, stateless ? nargs : nargs + 1, name)
        op = Operator(name, code, !stateless, sizeof(F), Base.datatype_alignment(F))
        Base.@lock operators_lock operators[(W, dev)] = op
    end
end

function compile_operator(W::Type, nargs::Int, name::String)
    config = CUDACore.compiler_config(device(); kernel=false, name)
    source = methodinstance(W, Tuple{ntuple(_ -> OpPtr, nargs)...})
    job = CompilerJob(source, config)
    JuliaContext() do ctx
        mod, meta = try
            GPUCompiler.compile(:llvm, job)
        catch err
            throws(err) || rethrow()
            throw(throwing_operator_error(err))
        end
        entry = meta.entry
        LLVM.name(entry) == name || error("operator was emitted as $(LLVM.name(entry))")

        # debug info only bloats the IR handed to HostJIT, which compiles without it
        LLVM.strip_debuginfo!(mod)

        convert(Vector{UInt8}, mod)
    end
end

# Device-side exceptions are reported through CUDA.jl's kernel state, which CUB's kernels
# do not have, so an operator that throws fails to compile. Even if it would, raising the
# exception makes the thread exit, which breaks the block-wide operations of CUB's
# scans and sorts (we saw illegal memory accesses). Report such operators clearly.
is_state_error(error) = error[3] == "julia.gpu.state_getter"
throws(err) = err isa InvalidIRError && all(is_state_error, err.errors)

# where an operator throws, without CUDA.jl's exception machinery and our wrapper
function throw_site((_, frames, _))
    runtime = dirname(pathof(CUDACore))
    last = findlast(frame -> startswith(String(frame.file), runtime), frames)
    last === nothing || (frames = frames[last+1:end])
    Base.filter(frame -> !(frame.file in (Symbol(@__FILE__), :unknown)), frames)
end

function throwing_operator_error(err::InvalidIRError)
    sites = Base.filter(!isempty, map(throw_site, err.errors))
    frames = isempty(sites) ? StackTraces.StackFrame[] : first(sites)

    msg = "CCCL operators cannot throw exceptions, but this one can"
    if isempty(frames)
        msg *= "."
    else
        msg *= ":\n" * join([" [$i] $frame" for (i, frame) in enumerate(frames)], '\n')
    end
    msg *= "\nAvoid operations that may throw, e.g., using `@inbounds`."
    return ArgumentError(msg)
end

# the `cccl_op_t` for an operator; `state` needs to be kept alive while it is used
function cccl_op(op::Operator, state::Union{Nothing,Base.RefValue}=nothing)
    cccl_op_t(op.stateful ? CCCL_STATEFUL : CCCL_STATELESS,
              reinterpret(Cstring, pointer(op.name)),
              reinterpret(Cstring, pointer(op.code)), length(op.code), CCCL_OP_LLVM_IR,
              op.size, op.alignment,
              state === nothing ? C_NULL : Base.unsafe_convert(Ptr{Cvoid}, state),
              C_NULL, C_NULL, 0, C_NULL)
end

# the state to pass along with an operator `f`
op_state(f) = isstateless(typeof(f)) ? nothing : Ref(f)
