

const KINDCONFIG = "kind_config"

# Lock states
const UNLOCKED = Int64(6)
const LOCKED = Int64(7)

# flag states
const IDLE = Int64(0)           # nothing is happening
const LOADING = Int64(1)        # host or device are transfering data
const HOST_CALL_BLOCKING = Int64(2)      # the host has handled hostcall
const HOST_CALL_NON_BLOCKING = Int64(3)      # host should handle hostcall
const HOST_HANDLING = Int64(4)  # host is handling hostcall
const HOST_DONE = Int64(5)  # host is handling hostcall


@inline packfoo(x, y)::Int64 = (Int64(x) << 32) | y
@inline unpackfoo(x)::Tuple{Int32, Int32} = ((x >>> 32) % Int32, (x & typemax(Int32)) % Int32)

@inline _ffs(x::Int32) = ccall("extern __nv_ffs", llvmcall, Int32, (Int32,), x)
@inline _ffs(x::UInt32) = ccall("extern __nv_ffs", llvmcall, Int32, (UInt32,), x)
@inline _ffs(x::Int64) = ccall("extern __nv_ffsll", llvmcall, Int32, (Int64,), x)
@inline _ffs(x::UInt64) = ccall("extern __nv_ffsll", llvmcall, Int32, (UInt64,), x)



"""
    align(value, bound=32)

Aligns value to bound

!!! Warning: Works only for bounds that are powers of 2!
"""
align(v, b=32) = (v + b-1) & ~(b-1);


"""
    KindConfig

Datastruct storing all required runtime information about the current area manager
"""
struct KindConfig
    stride::Int64
    area_size::Int64
    count::Int64
    kind::Int64
    area_ptr::Core.LLVMPtr{Int64,AS.Global}
end

struct Data
    a::Int32
    b::UInt32
    c::Int32
    d::Int32
end


@eval @inline manager_kind() =
    Base.llvmcall(
        $("""@$(KINDCONFIG) = weak addrspace($(AS.Constant)) externally_initialized global [$(sizeof(KindConfig)) x i8] zeroinitializer, align 8
             define i8 addrspace($(AS.Constant))* @entry() #0 {
                %ptr = getelementptr inbounds [$(sizeof(KindConfig)) x i8], [$(sizeof(KindConfig)) x i8] addrspace($(AS.Constant))* @$(KINDCONFIG), i64 0, i64 0
                %untyped_ptr = bitcast i8 addrspace($(AS.Constant))* %ptr to i8 addrspace($(AS.Constant))*
                ret i8 addrspace($(AS.Constant))* %untyped_ptr
            }
            attributes #0 = { alwaysinline }
          """, "entry"), LLVMPtr{KindConfig, AS.Constant}, Tuple{})


"""
    get_manager_kind()

Device function to get the current KindConfig
"""
function get_manager_kind()::KindConfig
    return unsafe_load(manager_kind())
end

abstract type AreaManager end

include("manager/simple.jl")
include("manager/warp.jl")

"""
    required_size(manager::AreaManager)

Function returning the expected minimum hostcall area size.
"""
required_size(manager::T) where {T <: AreaManager} = area_count(manager) * stride(manager)


"""
    kind_config(manager::AreaManager)

Function returning the current runtime KindConfig.
"""
function kind_config(manager::AreaManager, buffer::Mem.HostBuffer)
    ptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, buffer.ptr)
    KindConfig(stride(manager), manager.area_size, area_count(manager), kind(manager), ptr)
end


"""
    acquire_lock(kind::KindConfig)::(Int64, Ptr{Int64})

Device function acquiring a lock for the `kind` KindConfig
Returning an identifier (often an index) and a point for argument storing and return value gathering
"""
function acquire_lock(kindconfig::KindConfig, hostcall::Int64, blocking::Val{B})::Tuple{Data, Core.LLVMPtr{Int64,AS.Global}} where {B}
    if kindconfig.kind == kind(SimpleAreaManager)
        acquire_lock_impl(SimpleAreaManager, kindconfig, hostcall, blocking)
    elseif kindconfig.kind == kind(WarpAreaManager)
        acquire_lock_impl(WarpAreaManager, kindconfig, hostcall, blocking)
    else
        error("Unknown kindconfig")
    end
end


"""
    call_host_function(kind::KindConfig, index::Int64, hostcall::Int64)

Device function for invoking the hostmethod and waiting for identifier `index`.
"""
function call_host_function(kindconfig::KindConfig, index::Data, hostcall::Int64, blocking::Val{B}) where {B}
    if kindconfig.kind == kind(SimpleAreaManager)
        call_host_function_impl(SimpleAreaManager, kindconfig, index, hostcall, blocking)
    elseif kindconfig.kind == kind(WarpAreaManager)
        call_host_function_impl(WarpAreaManager, kindconfig, index, hostcall, blocking)
    else
        error("Unknown kindconfig")
    end
end


"""
    finish_function(kind::KindConfig, data::Data)

Device function for finishing a hostmethod, notifying the end of the invokation for identifier `data`.
"""
function finish_function(kindconfig::KindConfig, data::Data)
    if kindconfig.kind == kind(SimpleAreaManager)
        finish_function_impl(SimpleAreaManager, kindconfig, data)
    elseif kindconfig.kind == kind(WarpAreaManager)
        finish_function_impl(WarpAreaManager, kindconfig, data)
    else
        error("Unknown kindconfig")
    end
end


# Maps CuContext to big hostcall area
const hostcall_areas = Dict{CuContext, Mem.HostBuffer}()

"""
    assure_hostcall_area(ctx::CuContext, required::Int)::Mem.HostBuffer

Assures `ctx` has a HostBuffer of at least `required` size.
Returning that buffer.
"""
function assure_hostcall_area(ctx::CuContext, required)
    if !haskey(hostcall_areas, ctx) || sizeof(hostcall_areas[ctx]) < required
        haskey(hostcall_areas, ctx) && (println("Freeing"); Mem.free(hostcall_areas[ctx]))

        println("creating new hostcall area")
        hostcall_area = Mem.alloc(Mem.Host, required,
            Mem.HOSTALLOC_DEVICEMAP | Mem.HOSTALLOC_WRITECOMBINED)

        hostcall_areas[ctx] = hostcall_area
    end

    hostcall_areas[ctx]
end


has_hostcalls(ctx::CuContext) = haskey(hostcall_areas, ctx)
"""
    reset_hostcall_area!(mod::CuModule, manager::AreaManager)

This method is called just before a kernel is launched using this AreaManger.
Assuring a correct hostcall_area for `manager`.
Updating the KindConfig buffer with runtime config for `manager`.
"""
function reset_hostcall_area!(manager::AreaManager, mod::CuModule)::Ptr{Int64}
    hostcall_area = assure_hostcall_area(mod.ctx, required_size(manager))

    kind = kind_config(manager, hostcall_area)
    kind_global = CuGlobal{KindConfig}(mod, KINDCONFIG)
    kind_global[] = kind

    ptr = kind.area_ptr
    for i in 1:area_count(manager)
        unsafe_store!(ptr, UNLOCKED)
        unsafe_store!(ptr + 8, 0)
        unsafe_store!(ptr + 16, 0)
        ptr += stride(manager)
    end

    return reinterpret(Ptr{Int64}, kind.area_ptr)
end


"""
    check_area(manager::AreaManager, index::Int)::Vector{Ptr{Int64}}

Checks area `index` for open hostmethod calls.
There might be multiple open hostmethod calls related to as certain `index` (1-indexed).
"""
function check_area(manager::AreaManager, ptr::Ptr{Int64}, index::Int64)::Union{Nothing, Tuple{Int, Vector{Ptr{Int64}}}}
    ptr += stride(manager) * (index - 1)

    (state, hostcall) = volatile_load(reinterpret(Ptr{NTuple{2, Int64}}, ptr + 8))


    if (state == HOST_CALL_BLOCKING || state == HOST_CALL_NON_BLOCKING) && hostcall != 0
        unsafe_store!(ptr + 8, HOST_HANDLING)
        ptrs = areas_in(manager, ptr)
        return (hostcall, ptrs)
    end

    nothing
end


"""
    areas_in(::AreaManager, ptr::Ptr{Int64})::Vector{Ptr{Int64}}

Calculates all used hostmethod zones in `ptr`

! Expects the area to be in `HOST_CALL`
"""
areas_in


"""
    finish_area(manager::AreaManager, index::Int)

Finishes checking an area, notifying the device the host is done handling this function
"""
function finish_area(manager::T, ptr::Ptr{Int64}, index::Int64) where {T<:AreaManager}
    ptr += stride(manager) * (index - 1)
    volatile_store!(ptr + 8, HOST_DONE)
    volatile_store!(ptr+16, 0)
end


"""
    area_count(manager::AreaManager)::Int64

Method returning how many area's should be checked periodically
"""
area_count

function lock_area(ptr::Core.LLVMPtr{Int64,AS.Global})::Bool
    atomic_cas!(ptr, UNLOCKED, LOCKED) == UNLOCKED
end

function unlock_area(ptr::Core.LLVMPtr{Int64,AS.Global})::Bool
    v = atomic_xchg!(ptr, UNLOCKED)
    if v != LOCKED
        @cuprintln("Failed $v")
    end
    return v == LOCKED
end


function try_lock(cptr::Core.LLVMPtr{Int64,AS.Global})::Bool
    lock_area(cptr) || return false # Could not even lock this

    if unsafe_load(cptr+8) == IDLE # 'normal' case, the flag is on IDLE
        unsafe_store!(cptr+8, LOADING)
        return true
    end

    if unsafe_load(cptr + 16) == 0 # non-blocking case and host handled call
        unsafe_store!(cptr+8, LOADING)
        return true
    end

    unlock_area(cptr)

    return false
end
