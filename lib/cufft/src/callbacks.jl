# load and store callbacks written in Julia
#
# cuFFT links LTO callbacks into its kernels with nvJitLink when a plan is made, so that they
# are inlined into the transform. Each callback is a device function with a C prototype:
#
#     T load(void *dataIn, unsigned long long offset, void *callerInfo, void *sharedPointer);
#     void store(void *dataOut, unsigned long long offset, T element, void *callerInfo,
#                void *sharedPointer);
#
# cuFFT declares it in C++ code that it compiles with NVRTC, so the definition has to have
# the C++-mangled name, and the argument types that NVRTC uses for `float2`/`double2`, or
# nvJitLink can't inline it. We compile a Julia wrapper with a simple signature using
# GPUCompiler and libNVVM, the only compiler that generates LTO-IR, and add a shim with the
# exact C prototype that calls it.
#
# This is internal: the LazyArrays extension uses it to fuse broadcasts into transforms. A
# callback is a Julia function, `load(x, i)` returning the input element at linear index
# `i` of the input array `x`, or `store(y, i, v)` consuming element `v` of the output array
# `y`. It is converted like a kernel argument, and must not throw.

using GPUCompiler: GPUCompiler, CompilerConfig, CompilerJob, PTXCompilerTarget,
                   methodinstance
import LLVM
using LLVM: @dispose
using Core: LLVMPtr

const VoidPtr = LLVMPtr{UInt8,AS.Generic}

@inline load_aligned(::Type{T}, ptr::VoidPtr) where {T} =
    unsafe_load(reinterpret(LLVMPtr{T,AS.Generic}, ptr), 1, Val(Base.datatype_alignment(T)))
@inline store_aligned!(ptr::VoidPtr, val::T) where {T} =
    unsafe_store!(reinterpret(LLVMPtr{T,AS.Generic}, ptr), val, 1,
                  Val(Base.datatype_alignment(T)))

# what cuFFT's `callerInfo` points to: the converted callback, and the size of the array it
# accesses, as plans of different sizes share the compiled callback
struct CallbackState{F,N}
    f::F
    dims::Dims{N}
end

# `out[] = f(x, i)`, with `x` the array cuFFT loads from
struct LoadWrapper{F,S,N} end
@inline function (::LoadWrapper{F,S,N})(out::VoidPtr, data::VoidPtr, offset::UInt64,
                                        info::VoidPtr) where {F,S,N}
    state = load_aligned(CallbackState{F,N}, info)
    x = CuDeviceArray{S,N,AS.Generic}(reinterpret(LLVMPtr{S,AS.Generic}, data), state.dims)
    store_aligned!(out, convert(S, state.f(x, offset % Int + 1))::S)
    return
end

# `f(y, i, element[])`, with `y` the array cuFFT stores to
struct StoreWrapper{F,T,N} end
@inline function (::StoreWrapper{F,T,N})(element::VoidPtr, data::VoidPtr, offset::UInt64,
                                         info::VoidPtr) where {F,T,N}
    state = load_aligned(CallbackState{F,N}, info)
    y = CuDeviceArray{T,N,AS.Generic}(reinterpret(LLVMPtr{T,AS.Generic}, data), state.dims)
    state.f(y, offset % Int + 1, load_aligned(T, element))
    return
end

callback_type(::Type{<:LoadWrapper{<:Any,Float32}}) = CUFFT_CB_LD_REAL
callback_type(::Type{<:LoadWrapper{<:Any,Float64}}) = CUFFT_CB_LD_REAL_DOUBLE
callback_type(::Type{<:LoadWrapper{<:Any,ComplexF32}}) = CUFFT_CB_LD_COMPLEX
callback_type(::Type{<:LoadWrapper{<:Any,ComplexF64}}) = CUFFT_CB_LD_COMPLEX_DOUBLE
callback_type(::Type{<:StoreWrapper{<:Any,Float32}}) = CUFFT_CB_ST_REAL
callback_type(::Type{<:StoreWrapper{<:Any,Float64}}) = CUFFT_CB_ST_REAL_DOUBLE
callback_type(::Type{<:StoreWrapper{<:Any,ComplexF32}}) = CUFFT_CB_ST_COMPLEX
callback_type(::Type{<:StoreWrapper{<:Any,ComplexF64}}) = CUFFT_CB_ST_COMPLEX_DOUBLE

# the C type of an element, as NVRTC emits it: CUDA's vector types are named structs
c_type(::Type{Float32}) = LLVM.FloatType()
c_type(::Type{Float64}) = LLVM.DoubleType()
c_type(::Type{ComplexF32}) = named_struct("struct.float2", LLVM.FloatType())
c_type(::Type{ComplexF64}) = named_struct("struct.double2", LLVM.DoubleType())
function named_struct(name, elty)
    # the type has to have exactly this name, so reuse it if the context already has it
    ref = LLVM.API.LLVMGetTypeByName2(LLVM.context(), name)
    ref == C_NULL || return LLVM.StructType(ref)
    typ = LLVM.StructType(name)
    LLVM.elements!(typ, [elty, elty])
    return typ
end

# the Itanium mangling of an element type, and of the callback prototypes
mangle(::Type{Float32}) = "f"
mangle(::Type{Float64}) = "d"
mangle(::Type{ComplexF32}) = "6float2"
mangle(::Type{ComplexF64}) = "7double2"
mangled_name(::Type{<:LoadWrapper}, name) = "_Z$(length(name))$(name)PvyS_S_"
mangled_name(::Type{<:StoreWrapper{<:Any,T}}, name) where {T} =
    "_Z$(length(name))$(name)Pvy$(mangle(T))S_S_"

element_type(::Type{<:LoadWrapper{<:Any,S}}) where {S} = S
element_type(::Type{<:StoreWrapper{<:Any,T}}) where {T} = T

# add a function with the C prototype of a callback, calling the Julia wrapper `entry`
function add_shim!(mod::LLVM.Module, entry::LLVM.Function, W::Type, name::String)
    T = element_type(W)
    elty = c_type(T)
    ptr = LLVM.PointerType()
    i64 = LLVM.Int64Type()
    load = W <: LoadWrapper
    ft = if load
        LLVM.FunctionType(elty, [ptr, i64, ptr, ptr])
    else
        LLVM.FunctionType(LLVM.VoidType(), [ptr, i64, elty, ptr, ptr])
    end
    shim = LLVM.Function(mod, mangled_name(W, name), ft)

    entry_ft = LLVM.function_type(entry)
    @dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, LLVM.at_end(LLVM.BasicBlock(shim, "entry")))
        params = LLVM.parameters(shim)
        slot = LLVM.alloca!(builder, elty)
        if load
            LLVM.call!(builder, entry_ft, entry, [slot, params[1], params[2], params[3]])
            LLVM.ret!(builder, LLVM.load!(builder, elty, slot))
        else
            LLVM.store!(builder, params[3], slot)
            LLVM.call!(builder, entry_ft, entry, [slot, params[1], params[2], params[4]])
            LLVM.ret!(builder)
        end
    end

    # only the shim is part of the interface
    LLVM.linkage!(entry, LLVM.API.LLVMInternalLinkage)
    attrs = LLVM.function_attributes(entry)
    delete!(attrs, LLVM.EnumAttribute("noinline"))
    push!(attrs, LLVM.EnumAttribute("alwaysinline"))
    return shim
end

# a callback compiled to LTO-IR, defining the function `name` (before mangling). these are
# cached with the code instance of the wrapper, so that they are recompiled when a method
# the callback calls is redefined.
mutable struct CallbackResults
    name::Union{Nothing,String}
    code::Union{Nothing,Vector{UInt8}}
    CallbackResults() = new(nothing, nothing)
end

# a plan's load and store callbacks have to have different names
callback_name(W::Type) = W <: LoadWrapper ? "julia_cufft_load" : "julia_cufft_store"

# compiler parameters for callback wrappers, whose IR gets the shim with the C prototype
struct CallbackCompilerParams <: CUDACore.AbstractCUDACompilerParams
    sm::SMVersion
    ptx::VersionNumber
end
const CallbackJob = CompilerJob{PTXCompilerTarget,CallbackCompilerParams}

function GPUCompiler.finish_ir!(@nospecialize(job::CallbackJob), mod::LLVM.Module,
                                entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_ir!,
                   Tuple{CompilerJob{PTXCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)
    # (also called for deferred kernels compiled with the wrapper)
    W = job.source.specTypes.parameters[1]
    if W <: Union{LoadWrapper,StoreWrapper}
        add_shim!(mod, entry, W, callback_name(W))
    end
    return entry
end

# (taken around the check-and-compile sequence, as GPUCompiler doesn't serialize it)
const callback_compile_lock = ReentrantLock()

function callback_job(W::Type)
    # only libNVVM generates LTO-IR, whatever the code generator used for kernels.
    # the callback has no debug info, so strip it (after validation, for better errors).
    config = CUDACore.compiler_config(device(); kernel=false,
                                      name="$(callback_name(W))_julia", codegen=:nvvm)
    params = CallbackCompilerParams(config.params.sm, config.params.ptx)
    config = CompilerConfig(config; params, strip=true)
    source = methodinstance(W, Tuple{VoidPtr,VoidPtr,UInt64,VoidPtr})
    return CompilerJob(source, config)
end

# the outcome of compiling each wrapper, in the world it was compiled in. this saves
# setting up a compiler job for every execution, and remembers callbacks that don't compile.
const callback_memo = Dict{Tuple{Type,CuDevice},Tuple{UInt,Union{Nothing,CallbackResults}}}()

"""
    compiled_callback(W) -> Union{Nothing,CallbackResults}

Compile the callback wrapper `W` in the current world, or return earlier results that are
still valid. The results object identifies the compiled code: a different one is returned
after redefining a method the callback uses. Returns `nothing` for callbacks that cannot be
compiled, e.g. because they may throw.
"""
function compiled_callback(W::Type)
    key = (W, device())
    world = GPUCompiler.tls_world_age()
    Base.@lock callback_compile_lock begin
        memo = get(callback_memo, key, nothing)
        memo !== nothing && memo[1] == world && return memo[2]
        res = try
            compile_callback(W)
        catch err
            err isa CallbackError || rethrow()
            @debug "cuFFT callback does not compile" exception=(err, catch_backtrace())
            nothing
        end
        callback_memo[key] = (world, res)
        return res
    end
end

function compile_callback(W::Type)
    job = callback_job(W)
    GPUCompiler.run_compile_hook(job)
    res = GPUCompiler.cached_results(CallbackResults, job)
    if res === nothing || res.code === nothing
        code = emit_callback(job)
        res = GPUCompiler.cached_results(CallbackResults, job)::CallbackResults
        res.name = callback_name(W)
        res.code = code
    end
    return res
end

# a callback that cannot be linked into a transform
struct CallbackError <: Exception
    msg::String
end
Base.showerror(io::IO, err::CallbackError) = print(io, "CallbackError: ", err.msg)

# compile to LTO-IR, which the NVVM back-end emits as object code
function emit_callback(job::CompilerJob)
    try
        GPUCompiler.compile(:obj, job) do code, meta
            # kernels have their references to host objects patched when CUDA.jl loads
            # them, but cuFFT links the callback itself
            if !isempty(meta.relocations)
                targets = unique(map(relocation_target, meta.relocations.records))
                throw(CallbackError("the callback refers to Julia objects on the host: $(join(targets, ", "))"))
            end
            code
        end
    catch err
        # non-kernel functions can't report exceptions, as they lack the kernel state
        err isa GPUCompiler.InvalidIRError || rethrow()
        throw(CallbackError(sprint(showerror, err)))
    end
end

relocation_target(r::GPUCompiler.Relocation) = relocation_target(r.target)
relocation_target(t::GPUCompiler.JuliaValueRef) = sprint(show, t.value; context=:limit=>true)
relocation_target(t::GPUCompiler.CGlobalRef) = "the global `$(t.symbol)`"


## state

# a cuFFT handle with callbacks. `callerInfo` has to be configured when making the plan, so
# the device memory it points to belongs to the cached handle, and is updated every
# execution.
mutable struct CallbackHandle
    const handle::cufftHandle
    const load_info::Union{Nothing,CUDACore.DeviceMemory}
    const store_info::Union{Nothing,CUDACore.DeviceMemory}
    const workspace_size::Int
    # the last execution of the handle, which the next one waits for so that it doesn't
    # overwrite the state that one uses
    const done::CuEvent
end

Base.convert(::Type{cufftHandle}, h::CallbackHandle) = h.handle

state_type(::Type{<:LoadWrapper{F,<:Any,N}}) where {F,N} = CallbackState{F,N}
state_type(::Type{<:StoreWrapper{F,<:Any,N}}) where {F,N} = CallbackState{F,N}

# the wrapper type for a callback `f` accessing an array with element type `T` and `N`
# dimensions, and its compiled code. `nothing`s without a callback, and `missing` for one
# that cannot be used.
compiled_wrapper(::Type, ::Nothing, T, N) = (nothing, nothing)
function compiled_wrapper(::Type{W}, f, T, N) where {W}
    F = typeof(cudaconvert(f))
    isbitstype(F) || return missing
    cb = compiled_callback(W{F,T,N})
    cb === nothing && return missing
    return (W{F,T,N}, cb)
end

# set the callbacks of a newly created plan, before cuFFT makes it
function set_callbacks!(handle, callbacks, infos)
    for ((W, cb), info) in zip(callbacks, infos)
        W === nothing && continue
        caller_info = Ref(reinterpret(Ptr{Cvoid}, pointer(info)))
        GC.@preserve cb cufftXtSetJITCallback(handle, cb.name, cb.code, length(cb.code),
                                              callback_type(W), caller_info)
    end
end
