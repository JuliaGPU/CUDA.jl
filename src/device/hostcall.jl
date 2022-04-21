# device-side functionality for calling host methods
#
# MAJOR TODOs:
# - avoid deadlocks: the watcher thread isn't guaranteed to be scheduled when a kernel is
#   waiting for a response from the host when performing a block API call.
#   threads don't help here, since the Julia scheduler doesn't migrate work
#   (i.e. a run of the watcher) to an available thread (and we also want to support -t1):
#   https://github.com/JuliaGPU/CUDA.jl/pull/1140#issuecomment-916118046
#
# MINOR TODOs:
# - improve performance: currently takes around 2us per non-blocking uncontended hostcall.
#   all time spend in the atomic CAS, probably due to the PCI-E latency.
#   try using unified memory? ideally, avoiding atomics entirely is even better,
#   but that would require per-kernel and per-SM pools.
# - contended hostcalls are MUCH slower (try `@hostcall identity(nothing)` with more threads
#   that fit in the hostcall buffer).
# - 4K arg buffer per hostcall is wasteful, we could derive the size from the actual call.

export hostcall, @hostcall

@enum HostcallState::Int8 begin
    HOSTCALL_READY      # ready to receive a hostcall
    HOSTCALL_SUBMITTED  # params submitted, ready to process
    HOSTCALL_RETURNED   # host has stored return values (if any, else HOSTCALL_READY)
end

const HOSTCALL_BUFFER_SIZE = 4096

# GPU-compatible representation of a hostcall invocation
struct Hostcall
    state::HostcallState
    target::Int
    buffer::NTuple{HOSTCALL_BUFFER_SIZE, UInt8} # for parameters, and returned values

    # NOTE: the state and buffer fields should always be the first and last one respectively

    Hostcall(state, thread, block, target, buffer) =
        new(state, thread, block, target, buffer)
    Hostcall() = new(HOSTCALL_READY)
end

# list of called functions, represented in the Hostcall struct as an index into this list.
const hostcall_targets = []

"""
    hostcall(fun, rettyp, Tuple{argtyps...}, args...)

Call a function `fun` on the host, passing arguments `args` of types `argtyps`. The host
function returns `rettyp`, which is then returned by the hostcall. If `rettyp` is `Nothing`,
nothing is returned, and the hostcall will not have to wait on the CPU to finish the call.

!!! warning
    This interface is experimental, and might change without warning.
"""
@generated function hostcall(f::F, rettyp::Type{T}, ::Type{U}, args...) where {F,T,U}
    # register the target
    sig = Tuple{F, U.parameters...}
    push!(hostcall_targets, (; sig, rettyp=T))
    index = length(hostcall_targets)

    # perform ccall-like argument conversion (cconvert |> unsafe_convert)
    argtypes = Type[U.parameters...]
    convert_argument_exprs(argtypes, args, :perform_hostcall, :f, :rettyp, index)
end

@inline function perform_hostcall(f, rettyp::Type{T}, index::Int, args...) where {T}
    # NOTE: this function has been carefully implemented to avoid throwing any exception,
    #       even trivial ones that are optimized away (e.g. by calling `UInt32(0)`).
    #       this is because hostcall is used to implement throw_* functions,
    #       and we otherwise run into recursion during inference.
    #       debug by enabling inference remarks and looking for:
    #       "compilation of Core.throw_*(...): Bounded recursion detected"

    # XXX: timeouts to prevent deadlocks?
    mask = active_mask()
    slots = popc(mask)
    leader = ffs(mask)

    # reserve the amount of hostcall slots this warp needs
    head0 = 0x00000000
    if laneid() == leader
        pointers = hostcall_pointers()
        pointers_ptr = pointer(pointers)
        pointers_align = 4
        #@inbounds head0, tail0 = pointers[1], pointers[2]
        head0 = unsafe_load(pointers_ptr, 1, Val(pointers_align))
        tail0 = unsafe_load(pointers_ptr, 2, Val(pointers_align))
        while true
            if ring_space(head0, tail0, HOSTCALL_POOL_SIZE) >= slots
                cmp = head0
                new_head0 = head0 + slots   # clamped to valid range below
                head0 = atomic_cas!(pointers_ptr, cmp, new_head0)
                (head0 == cmp) && break
            else
                # wait for the CPU to process items
                compute_capability() >= sv"7.0" && nanosleep(1024u32)
                #@inbounds tail0 = pointers[2]
                tail0 = unsafe_load(pointers_ptr, 2, Val(pointers_align))
            end
        end
    end

    sync_warp(mask)

    # get our own slot
    base0 = shfl_sync(mask, head0, leader)
    idx0 = popc(mask & ((0x00000001 << (laneid() - 0x1)) - 0x1))
    slot = (base0 + idx0) & (HOSTCALL_POOL_SIZE - 0x1) + 0x1

    # wait for the slot to be available (another thread may still be processing returned values)
    while hostcall_state(slot) != HOSTCALL_READY
        compute_capability() >= sv"7.0" && nanosleep(1024u32)
    end

    # submit the hostcall
    hostcall_target!(index, slot)
    write_hostcall_arguments(hostcall_buffer_ptr(slot), f, args...)
    hostcall_state!(HOSTCALL_SUBMITTED, slot)

    if rettyp === Nothing
        # non-blocking hostcall; let's just continue
        rv = nothing
    else
        # wait for the last returned value (implying all preceding ones are ready too)
        if idx0 + 0x1 == slots
            while hostcall_state(slot) == HOSTCALL_SUBMITTED
                compute_capability() >= sv"7.0" && nanosleep(1024u32)
            end
        end

        sync_warp(mask)

        if hostcall_state(slot) == HOSTCALL_READY
            # something went wrong... let's bail out
            trap()
        end

        rv = unsafe_load(reinterpret(LLVMPtr{T,AS.Global}, hostcall_buffer_ptr(slot)), 1,
                         Val(Base.datatype_alignment(T)))

        # release the parameters
        hostcall_state!(HOSTCALL_READY, slot)
    end

    # NOTE: the flag _needs_ to be set to READY here, either by the CPU or the GPU, because
    #       otherwise the CPU could try to access the hostcall object before it has been
    #       fully initialized (but after the tail pointer has been bumped).

    return rv::T
end

# generated helper to efficiently write hostcall argument, without iterating at run time.
@inline @generated function write_hostcall_arguments(ptr, args...)
    ex = quote end

    # NOTE: we use the same storage convention as dynamic parallelism
    last_offset = 0
    for i in 1:length(args)
        T = args[i]
        sz = sizeof(T)
        if sz > 0
            align = Base.datatype_alignment(T)
            offset = Base.cld(last_offset, align) * align
            last_offset = offset + sz
            if last_offset > HOSTCALL_BUFFER_SIZE
                # buffer overrun; bail out, the CPU will warn about this
                break
            end
            push!(ex.args, :(
                unsafe_store!(reinterpret(LLVMPtr{$T,AS.Global}, ptr+$offset),
                              args[$i], 1, Val($align))
            ))
        end
    end

    ex
end


## convenience macro

"""
    @hostcall fun([args...])
    @hostcall fun([args...])::T

Call the function `fun` on the host, passing `args`. The return typeof the function is
inferred. If this fails, the return type may be specified explicitly using the `::T` syntax.

See also: [`hostcall`](@ref)

!!! warning
    This interface is experimental, and might change without warning.
"""
macro hostcall(ex)
    # check if the return type is specified
    if Meta.isexpr(ex, :(::))
        ex, rettyp = ex.args
    else
        rettyp = nothing
    end

    # decode the call
    @assert Meta.isexpr(ex, :call)
    f, args... = ex.args

    # forward to a generated function to figure out the argument types
    esc(quote
        $emit_hostcall($f, $rettyp, $(args...))
    end)
end

@generated function emit_hostcall(f::F, retspec::T, args...) where {F, T}
    argtyps = Tuple{args...}

    # determine the return type
    if retspec <: Type
        # the user has provided the type
        rettyp = retspec.parameters[1]
    elseif isdefined(F, :instance)
        # check with inference
        rettyp = Core.Compiler.return_type(F.instance, argtyps)
        if rettyp === Union{}
            tn = F.name::Core.TypeName
            fn = isdefined(tn, :mt) ? tn.mt.name : string(F)
            Core.println("WARNING: @hostcall could not deduce return type of '$fn($(args...))'; try annotating the call instead")
            rettyp = Nothing
        end
    else
        Core.println("WARNING: @hostcall cannot deduce return type closures; annotating the call instead")
        rettyp = Nothing
    end

    quote
        hostcall(f, $rettyp, $argtyps, args...)
    end
end
