include("hostcall/notification.jl")
include("hostcall/area_manager.jl")
include("hostcall/hostref.jl")
include("hostcall/timer.jl")
include("hostcall/poller.jl")

ret_offset() = 16
"""
    handle_hostcall(llvmptr::Core.LLVMPtr{Int64,AS.Global})

Host side function that checks and executes outstanding hostmethods.
Checking only one index.
"""
function handle_hostcall(manager::AreaManager, area::Ptr{Int64}, policy::NotificationPolicy, policy_area::Ptr{Int64}, index::Int64)::Vector{Int64}
    out = Int64[]

    for (hostcall, ptrs, area_index) in check_area(manager, area, policy, policy_area, index)
        push!(out, hostcall)
        try
            ## Handle hostcall
            handle_hostcalls(Val(hostcall), manager, ptrs, area, area_index)
        catch e
            println("ERROR ERROR hostcall $(hostcall)")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println(stdout)
            end

            finish_area(manager, area, area_index)
        end
    end

    return out
end


"""
    handle_hostcalls(::Val{N}, manager::AreaManager, ptrs::Ptr{UInt8}, area::Ptr{Int64}, index::Int64)

    TODO
Executes hostmethod with id N using hostcall area at `ptr`

This is the catch all function, but specialized functions are generated inside `@cpu`.
Specialized functions are aliased to `exec_hostmethode` with particular types.
"""
function handle_hostcalls(::Val{N}, manager::AreaManager, ptrs::Vector{Ptr{Int64}}, area::Ptr{Int64}, index::Int64) where {N}
    println("Syscall $N not supported")
end


"""
    exec_hostmethodes(::Type{A}, ::Type{R}, func::Function, ptr::Ptr{UInt8})

Downloads and invconverts arguments in hostcall area
Actually call the hostmethod
Set converted return argument in hostcall area
"""
function exec_hostmethodes(::Type{A}, ::Type{R}, manager::AreaManager, func::Function, ptrs::Vector{Ptr{Int64}}, area::Ptr{Int64}, index::Int64, blocking::Bool) where {A, R}
    function load_arg(ptr::Ptr{Int64})
        arg_tuple_ptr = reinterpret(Ptr{A}, ptr)
        arg_tuple = unsafe_load(arg_tuple_ptr)

        args = map(invcudaconvert, arg_tuple)

        (args, ptr)
    end

    args = map(load_arg, ptrs)

    if !blocking
        finish_area(manager, area, index)
    end

    function exec_f((args, ptr))
        ret_v = func(args...)

        if blocking
            ret = cudaconvert(ret_v)
            ret_ptr = reinterpret(Ptr{R}, ptr + ret_offset())
            unsafe_store!(ret_ptr, ret)
        end
    end

    foreach(exec_f, args)

    if blocking
        finish_area(manager, area, index)
    end
end


"""
    call_hostcall(::Type{R}, n, args::A)

Device side function to call a hostmethod. This function is invoked with `@cpu`.
`R` is the return argument type.
`n` is the hostcall id, a number to identify the required hostmethod to execute.
"""
@inline function call_hostcall(::Type{R}, n, args::A, blocking::Val{B}) where {R, A, B}
    # Get the area_manager of current execution
    kind_config = get_manager_kind()

    # Acquire lock
    (index, llvmptr) = acquire_lock(kind_config, n)
    while reinterpret(Int64, llvmptr) == 0
        (index, llvmptr) = acquire_lock(kind_config, n)
    end

    # Store arguments
    args_ptr = reinterpret(Ptr{A}, llvmptr)
    unsafe_store!(args_ptr, args)

    # Call host
    call_host_function(kind_config, index, n, blocking)

    if B
        # Get return value
        ret_ptr = reinterpret(Ptr{R}, llvmptr + ret_offset())
        local ret = unsafe_load(ret_ptr)

        # Finish method
        finish_function(kind_config, index)

        # Return
        ret
    else
        nothing
    end
end


"""
    TypeCache{T, I}

Struct to relate T values to I index (mostly Int64, Int32)
Same T values are mapped to same I index
"""
struct TypeCache{T, I}
    stash::Dict{T, I}
    vec::Vector{T}
end


function type_to_int!(cache::TypeCache{T, I}, type::T) where {T, I}
    if haskey(cache.stash, type)
        return cache.stash[type]
    else
        push!(cache.vec, type)
        cache.stash[type] = length(cache.vec)
    end
end


int_to_type(cache::TypeCache{T, I}, index::I) where {T, I} = cache.vec[index]


""" Struct to keep track of all compiled hostmethods """
# This is not really used at it's best
const cpufunctions = TypeCache{Tuple{Symbol, Expr, Bool}, Int64}(Dict(), Vector())


"""
    @cpu [kwargs...] func(args...)

High-level interface for call a methode on the host from a device. The `@cpu` macro should prefix a call,
with `func` a callable function or object. Arguments from the device are converted using `invcudaconvert`
on the host before calling the function. Any return arguments are converted using `cudaconvert` before
returning to the device.

The keyword `types` is required and uses a Tuple of types. The first type is the return argument and
the others are the argument types (both as seen on device side).
"""
macro cpu(ex...)
    # destructure the `@cpu` expression
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the cpu call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @cuda should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    macro_kwargs, other_kwargs = split_kwargs(kwargs, [:types, :blocking, :async, :gran_gather, :gran_scatter])

    types = nothing
    blocking = true
    async = false

    gran_gather = nothing
    gran_scatter = nothing

    arg_c = length(args) + 1 # number of arguments + return type

    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key == :types
            types = eval(val)::NTuple{arg_c, DataType}
        elseif key == :blocking
            isa(val, Bool) || throw(ArgumentError("`blocking` keyword argument to @cpu should be a constant value"))
            blocking = val::Bool
        elseif key == :async
            isa(val, Bool) || throw(ArgumentError("`async` keyword argument to @cpu should be a constant value"))
            async = val::Bool
        elseif key == :gran_gather
            gran_gather = val
        elseif key == :gran_scatter
            gran_scatter = val
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end

    if types === nothing
        throw(ArgumentError("'types' keyword argument is required (for now), with 1 tuple argument"))
    end

    if gran_gather !== nothing && gran_scatter === nothing
        throw(ArgumentError("When supplying 'gather' function please also supply 'scatter' function"))
    end

    if gran_scatter !== nothing && gran_gather === nothing
        throw(ArgumentError("When supplying 'scatter' function please also supply 'gather' function"))
    end


    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # make sure this exists
    # To be safe this should just increment
    # this has multiple problems
    indx = type_to_int!(cpufunctions, (f, quote types end, blocking))

    println("hostcall $indx -> $f")
    # remember this module
    caller_module = __module__

    # Convert (Int, Int,) -> Tuple{Int, Int} which is the type of the arguments
    types_type_quote = :(Tuple{$(types[2:end]...)})

    # handle_hostcall function that is called from handle_hostcall(ctx::CuContext)
    new_fn = quote
        handle_hostcalls(::Val{$indx}, manager::AreaManager, ptrs::Vector{Ptr{Int64}}, area::Ptr{Int64}, index::Int64) = exec_hostmethodes($types_type_quote, $(types[1]), manager, $caller_module.$f, ptrs, area, index, $blocking)
    end


    # Put function in julia space
    eval(new_fn)

    # Convert to correct arguments
    args_tuple = Expr(:tuple, args...)

    if gran_gather !== nothing
        call_cpu = quote
            CUDA.gather_scatter($args_tuple, $(types[1]), $gran_gather, $gran_scatter, v -> CUDA.call_hostcall($(types[1]), $indx, v, Val($blocking)))
        end
    else
        call_cpu = quote
            CUDA.call_hostcall($(types[1]), $indx, $args_tuple, Val($blocking))
        end
    end

    return esc(call_cpu)
end


"""
    macro dotimes f times start

Macro to unroll `times` itself with inner state `start`.

The argument is inside tuple to prevent `eval` from evaling that Expr
f is a function (Expr,) -> Expr.
"""
macro dotime(f, times, current=(quote end))
    for i in 1:times
        inner = (current, nothing,)
        current = eval(quote $f($inner) end)
    end
    return esc(current)
end

_popc(x::UInt32) = ccall("extern __nv_popc", llvmcall, UInt32, (UInt32,), x)
"""
    gather_scatter(value::T, returntype::Type{R}, gather_f, scatter_f, f)::R

    Execute gather scatter algorithm at warp level.

    gather_f  :: T -> T -> (S, T) (unite 2 arguments into 1 argument, leaving behind some state)
    scatter_f :: S -> R -> (R, R) (scatter 1 return value with some created state into 2 return values)
    f         :: T -> R (execute operant on big value)

    ! Important !
    This uses dynamic shared memory, please provide your kernel invocation with enough shared memory.
    Enough: sizeof(T) * threads + sizeof(R) * threads

    Expectation:
      (state, large_arg) = gather_f(arg1, arg2)
      (f(arg1), f(arg2)) == scatter_f(state, f(large_arg))

      # T -> T -> (S, T)
      add_with_state(v1, v2) = (v1, v1+v2)
      # S -> R -> (R, R)
      get_with_state(state, t) = (state * 2, t-(state*2))

      vv = CUDA.gather_scatter(threadx, Int64, add_with_state, get_with_state, c -> c*2)


    ```julia-REPL
    julia> add_with_state(v1, v2) = (v1, v1+v2) # T -> T -> (S, T)
    julia> get_with_state(state, t) = (state * 2, t-(state*2)) # S -> R -> (R, R)
    julia> function kernel()
        id = threadIdx().x
        id2 = CUDA.gather_scatter(id, Int64, add_with_state, get_with_state, c -> c*2)
        @cuprintln("\$id * 2 == \$id2")
        return
    end
    julia> @cuda threads=5 shmem=5*2*sizeof(Int64) kernel()
    1 * 2 = 2
    2 * 2 = 4
    3 * 2 = 6
    4 * 2 = 8
    5 * 2 = 10
    ```
"""
@inline function gather_scatter(v::T, ::Type{R}, gather, scatter, f)::R where {T, R}
    sync_warp() # This is a bug (shouldn't be needed)

    # TODO: make memory smaller by overlapping, either sm_t or sm_r is used at once
    # get for this T, a slice of shared memory to gather arguments
    sm_t = @cuDynamicSharedMem(T, 32, div(threadIdx().x - 1, 32) * 32 * sizeof(T))
    # get behind all T slices a slice of shared memory to scatter return values
    sm_r = @cuDynamicSharedMem(R, 32, blockDim().x * sizeof(T) + div(threadIdx().x - 1, 32) * 32 * sizeof(R))

    threadx = (blockIdx().x-1) * align(blockDim().x) + threadIdx().x - 1
    laneid = threadx % 32
    mask = vote_ballot(true)
    lane_mask = UInt32((1 << laneid) - 1)
    rank = _popc(mask & lane_mask)

    popc = _popc(mask) # how many threads want to gather and scatter

    # initialize gather
    @inbounds sm_t[rank+1] = v
    sync_warp()

    i = 1 # current warp level merge
    mask1 = mask # current mask of sync threads

    """
    # means good && above < popc
    o used value, but not itself
    . means unused value

rank
000  #-#--#----#
001  o/. /.   /.
010  #-o/ .  / .
011  o/.  . /  .
100  #-#--o/   .
101  o/. /.    .
110  #-o/ .    .
111  o/.  .    .

    """

    @dotime(c -> begin
        @gensym state last_mask above #something like scope variables
        quote
            # so gather rank and above togather into rank, this is a check for not fully filled warps etc
            $above = rank | i
            # rank has a zero at place i
            good = i & rank == 0

            # the new mask of this iteration
            mask1 = vote_ballot_sync(mask1, good)
            # store in local scope
            $last_mask = mask1

            if good # My turn to be useful
                if $above < popc
                    ($state, v) = gather(sm_t[rank+1], sm_t[$above+1])
                    @inbounds sm_t[rank+1] = v
                end

                i = i << 1
                sync_warp($last_mask)

                $(c[1])

                sync_warp($last_mask)

                # if this is not the case, the expected value is already at sm_r[rank+1]
                if $above < popc
                    (v1, v2) = scatter($state, sm_r[rank+1])

                    @inbounds sm_r[rank+1] = v1
                    @inbounds sm_r[$above+1] = v2
                end
            end
        end

    end, 5, start=(@inbounds sm_r[1] = f(sm_t[1]))) # 2 ** 5 == 32

    sync_warp(mask)

    return sm_r[rank+1]
end


@generated function volatile_store!(ptr::Ptr{T}, value, index=1) where T
    JuliaContext() do ctx
        ptr_type = convert(LLVMType, Ptr{T}, ctx)
        lt = convert(LLVMType, T, ctx)

        ir = """
            %ptr = inttoptr $ptr_type %0 to $lt*
            store volatile $lt %1, $lt* %ptr
            ret void
            """
        :(Core.Intrinsics.llvmcall($ir, Cvoid, Tuple{$(Ptr{T}), $T}, ptr + (index - 1) * sizeof(T), convert(T, value)))
    end
end


@generated function volatile_load(ptr::Ptr{T}, index=1) where T
    JuliaContext() do ctx
        ptr_type = convert(LLVMType, Ptr{T}, ctx)
        lt = convert(LLVMType, T, ctx)

        ir = """
            %ptr = inttoptr $ptr_type %0 to $lt*
            %value = load volatile $lt, $lt* %ptr
            ret $lt %value
            """

        :(Base.llvmcall($ir, T, Tuple{Ptr{T}}, ptr + (index - 1) * sizeof(T)))
    end
end
