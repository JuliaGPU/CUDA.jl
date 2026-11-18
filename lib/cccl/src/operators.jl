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
const operators_lock = ReentrantLock()
const operator_counter = Threads.Atomic{Int}(0)

"""
    CCCL.operator(f, A, B, R) -> Operator

Compile the binary function `f`, called with arguments of type `A` and `B` and returning a
value convertible to `R`, to an operator for use with CCCL algorithms.
"""
function operator(f::F, A::Type, B::Type, R::Type) where {F}
    isbitstype(F) ||
        throw(ArgumentError("CCCL operators must be bits values; $F captures non-bits state"))
    stateless = isstateless(F)
    W = stateless ? StatelessOp{F,A,B,R} : StatefulOp{F,A,B,R}
    dev = device()
    Base.@lock operators_lock begin
        get!(operators, (W, dev)) do
            name = "cccl_julia_op_$(Threads.atomic_add!(operator_counter, 1))"
            code = compile_operator(W, stateless ? 3 : 4, name)
            Operator(name, code, !stateless, sizeof(F), Base.datatype_alignment(F))
        end
    end
end

function compile_operator(W::Type, nargs::Int, name::String)
    config = CUDACore.compiler_config(device(); kernel=false, name)
    source = methodinstance(W, Tuple{ntuple(_ -> OpPtr, nargs)...})
    job = CompilerJob(source, config)
    JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry = meta.entry
        LLVM.name(entry) == name || error("operator was emitted as $(LLVM.name(entry))")

        # debug info only bloats the IR handed to HostJIT, which compiles without it
        LLVM.strip_debuginfo!(mod)

        convert(Vector{UInt8}, mod)
    end
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
