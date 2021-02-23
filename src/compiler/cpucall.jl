const CPUCALL_AREA = "cpucall_area"
const JULIA_CPUCALL_AREA = "julia_cpucall_area"


function wait_and_kill_watcher(e::CuEvent, ctx::CuContext)
    println("Start the watcher!")

    try
        while !query(e)
            handle_cpucall(ctx)
            yield()
        end
    catch e
        println("Failed $e")
    end
    println("Killed the watcher!")
end

cpucall_area_size() = sizeof(UInt8) * 1024 * 1024

function emit_cpucall_area!(mod::LLVM.Module)
    @assert !haskey(globals(mod), CPUCALL_AREA)
    ctx = LLVM.context(mod)

    # add the global variable
    T_ptr = convert(LLVMType, Ptr{Cvoid}, ctx)
    gv = GlobalVariable(mod, T_ptr, CPUCALL_AREA)
    initializer!(gv, LLVM.ConstantInt(T_ptr, 0))
    linkage!(gv, LLVM.API.LLVMWeakAnyLinkage)
    extinit!(gv, true)
    set_used!(mod, gv)

    if haskey(functions(mod), JULIA_CPUCALL_AREA)
        buf_getter = functions(mod)[JULIA_CPUCALL_AREA]
        @assert return_type(eltype(llvmtype(buf_getter))) == eltype(llvmtype(gv))

        # find uses
        worklist = Vector{LLVM.CallInst}()
        for use in uses(buf_getter)
            call = user(use)::LLVM.CallInst
            push!(worklist, call)
        end

        # replace uses by a load from the global variable
        for call in worklist
            Builder(ctx) do builder
                position!(builder, call)
                ptr = load!(builder, gv)
                replace_uses!(call, ptr)
            end
            unsafe_delete!(LLVM.parent(call), call)
        end
    end
end

const cpucall_areas = Dict{CuContext, Mem.HostBuffer}()

dump_memory(ty=UInt8) = dump_memory{UInt8}(ty)
dump_memory(ty::Type{T}, size=cpucall_area_size() ÷ sizeof(ty), ctx::CuContext=context()) where {T} = dump_memory(ty, size_in_bytes, ctx)
function dump_memory(::Type{T}, size::Int64, ctx::CuContext=context()) where {T}
    ptr    = convert(CuPtr{T}, cpucall_areas[ctx])
    cuarray = unsafe_wrap(CuArray{T}, ptr, size)
    println("Dump $cuarray")
end

function reset_cpucall_area!(ctx::CuContext)
    println("resetting")
    dump_memory(Int, 10, ctx)
    ptr    = convert(CuPtr{UInt8}, cpucall_areas[ctx])
    cuarray = unsafe_wrap(CuArray{UInt8}, ptr, cpucall_area_size())

    # Reset array to 0
    fill!(cuarray, 0)
end

function create_cpucall_area!(mod::CuModule)
    flag_ptr = CuGlobal{Ptr{Cvoid}}(mod, CPUCALL_AREA)
    cpucall_area = get!(cpucall_areas, mod.ctx,
        Mem.alloc(Mem.Host, cpucall_area_size(), Mem.HOSTALLOC_DEVICEMAP | Mem.HOSTREGISTER_DEVICEMAP))
    flag_ptr[] = reinterpret(Ptr{Cvoid}, convert(CuPtr{Cvoid}, cpucall_area))

    unsafe_store!(convert(Ptr{Int64}, cpucall_area), 0)
    unsafe_store!(convert(Ptr{Int64}, cpucall_area), 0, 2)

    return
end


function handle_cpucall(ctx::CuContext)

    ptr    = convert(Ptr{Int64}, cpucall_areas[ctx])
    ptr_u8 = convert(Ptr{UInt8}, cpucall_areas[ctx])

    llvmptr    = reinterpret(Core.LLVMPtr{Int64,AS.Global}, ptr)

    flag = unsafe_load(ptr)
    if flag != CPU_CALL
        return
    end

    # Notify CPU is handling this cpucall
    unsafe_store!(ptr, CPU_HANDLING)

    # Fetch this cpucall
    cpucall = unsafe_load(ptr+8)

    println("handling syscall $cpucall")
    try
        ## Handle cpucall
        handle_cpucall(Val(cpucall), ptr_u8+16)
    catch e
        println("ERROR ERROR")
        println(e)
    end

    # Notify end
    unsafe_store!(ptr, CPU_DONE)
end

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


function handle_cpucall(::Val{N}, kwargs...) where {N}
    println("Syscall $N not yet supported")
end


const cpufunctions = TypeCache{Symbol, Int64}(Dict(), Vector())
const type_cache = TypeCache{DataType, Int32}(Dict(), Vector())

macro cpu(ex...)
    # destructure the `@cpu` expression
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the cpu call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @cuda should be a function call"))
    f = call.args[1]
    args = call.args[2:end]


    types_kwargs, other_kwargs = split_kwargs(kwargs, [:types])

    if length(types_kwargs) != 1
        throw(ArgumentError("'types' keyword argument is required (for now), with 1 tuple argument"))
    end

    _,val = types_kwargs[1].args

    arg_c = length(args) + 1 # number of arguments
    types = eval(val)::NTuple{arg_c, DataType} # types of arguments

    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end


    # make sure this exists
    indx = type_to_int!(cpufunctions, f)

    # remember this module
    caller_module = __module__


    # handle_cpucall function that is called from handle_cpucall(ctx::CuContext)
    new_fn = quote
        function handle_cpucall(::Val{$indx}, ptr::Ptr{UInt8})
            local ar = []
            local func = $caller_module.$f

            ptr_64 = reinterpret(Core.LLVMPtr{Int32,AS.Global}, ptr)
            vec = CuDeviceArray{Int32}(16, ptr_64) # this 16 is quite strange

            # Get the number of arguments of the function
            arg_c = vec[1]
            types = []

            # Get the types of the function
            for x in 1:arg_c
                type_int = vec[x+1]
                push!(types, int_to_type(type_cache, type_int))
            end

            # Calculate offset of first argument
            offset = 256 + ((arg_c * sizeof(Int32)) ÷ 256) * 256
            for type in types[2:end]
                # Get CuDeviceArray to argument
                ptr_tt = reinterpret(Core.LLVMPtr{type,AS.Global}, ptr + offset)
                vec = CuDeviceArray{type}(1, ptr_tt)

                # Get argument
                push!(ar, vec[1])

                # Adjust offset
                offset += sizeof(type)
            end

            # Actually call function
            ret = func(ar...)

            # Calculate offset of return value
            offset = 256 + ((arg_c * sizeof(Int32)) ÷ 256) * 256
            ret_ptr = reinterpret(Core.LLVMPtr{types[1], AS.Global}, ptr + offset)
            CuDeviceArray{types[1]}(16, ret_ptr)[1] = ret
        end
    end

    # Put function in julia space
    eval(new_fn)

    @gensym ptr_ident

    # payload_code start, set number of arguments
    payload_code = quote
        ptr_64 = reinterpret(Core.LLVMPtr{Int32,AS.Global}, $ptr_ident)
        vec = CuDeviceArray{Int32}(16, ptr_64)
        vec[1] = $arg_c
    end

    # Get all types of arguments
    for (i, type) in enumerate(types)
        type_index = type_to_int!(type_cache, type)
        push!(payload_code.args, quote
            vec[$(i+1)] = $type_index
        end)
    end

    # Calculate offset of first argument
    offset = 256 + ((arg_c * sizeof(Int32)) ÷ 256) * 256

    for (arg, type) in zip(args, types[2:end])
        # Set the argument at the correct offset in shared memory in cpu call area
        push!(payload_code.args, quote
            CuDeviceArray{$type}(16, reinterpret(Core.LLVMPtr{$type, AS.Global}, $ptr_ident + $offset))[1] = $arg
        end)

        # Adjust offset with alignment
        offset += sizeof(type)
    end

    # calculate return value drop site
    offset = 256 + ((arg_c * sizeof(Int32)) ÷ 256) * 256
    ret_type = types[1]
    ret_size = sizeof(ret_type)

    call_cpu = quote
        CUDA.call_syscall($indx, ptr -> CuDeviceArray{$ret_type}(16, reinterpret(Core.LLVMPtr{$ret_type, AS.Global}, ptr + $offset))[1]) do $ptr_ident
            $payload_code
            return
        end
    end

    return call_cpu
end
