const KINDCONFIG = "kind_config"

# flag states
const IDLE = Int64(0)           # nothing is happening
const HOST_DONE = Int64(1)      # the host has handled hostcall
const LOADING = Int64(2)        # host or device are transfering data
const HOST_CALL = Int64(3)      # host should handle hostcall
const HOST_HANDLING = Int64(4)  # host is handling hostcall

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
    stride::UInt64
    count::UInt64
    kind::Int64
    area_ptr::Core.LLVMPtr{Int64,AS.Global}
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
    # ptr = reinterpret(Ptr{KindConfig}, manager_kind())
    return unsafe_load(manager_kind())
end



abstract type AreaManager end

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
    KindConfig(stride(manager), area_count(manager), kind(manager), ptr)
end


"""
    SimpleAreaManager(area_count, area_size)

Simple AreaManager that makes threads fight for locks.
There are `area_count` hostcall areas.
"""
struct SimpleAreaManager <: AreaManager
    area_count::Int
    area_size::Int
end
stride(manager::SimpleAreaManager) = align(manager.area_size + 2 * sizeof(Int64))
kind(::SimpleAreaManager) = 0


"""
    WarpAreaManager(warp_area_count, area_size, warp_size=32)

Less simple AreaManager that locks an area per warp when invoked, reducing congestion.
Uses Opportunistic Warp-level Programming to achieve this.
"""
struct WarpAreaManager <: AreaManager
    warp_area_count::Int
    area_size::Int
    warp_size::Int
end
stride(manager::WarpAreaManager) = align(manager.area_size * manager.warp_size + 3 * sizeof(Int64))
kind(::WarpAreaManager) = 1


"""
    acquire_lock(kind::KindConfig)::(UInt64, Ptr{Int64})

Device function acquiring a lock for the `kind` KindConfig
Returning an identifier (often an index) and a point for argument storing and return value gathering
"""
function acquire_lock(kind::KindConfig)::Tuple{UInt64, Core.LLVMPtr{Int64,AS.Global}}
    if kind.kind == 0 # SimpleAreaManager
        ptr = kind.area_ptr
        stride = kind.stride
        count = kind.count

        i = threadIdx().x - 1

        while atomic_cas!(ptr + (i % count) * stride, IDLE, LOADING) != IDLE
            nanosleep(UInt32(16))
            i += 1
        end

        return (i%count, ptr + (i % count) * stride + 16)
    else # WarpAreaManager
        # TODO
        (0, kind.area_ptr)
    end
end


"""
    call_host_function(kind::KindConfig, index::UInt64, hostcall::Int64)

Device function for invoking the hostmethod and waiting for identifier `index`.
"""
function call_host_function(kind::KindConfig, index::UInt64, hostcall::Int64)
    ptr = reinterpret(Ptr{Int64}, kind.area_ptr + index * kind.stride)
    unsafe_store!(ptr + 8, hostcall)
    threadfence()
    unsafe_store!(ptr, HOST_CALL)

    while volatile_load(ptr + 8) != 0
        nanosleep(UInt32(16))
        threadfence()
    end

    unsafe_store!(ptr, LOADING)
end


"""
    finish_function(kind::KindConfig, index::UInt64)

Device function for finishing a hostmethod, notifying the end of the invokation for identifier `index`.
"""
function finish_function(kind::KindConfig, index::UInt64)
    ptr = reinterpret(Ptr{Int64}, kind.area_ptr + index * kind.stride)
    unsafe_store!(ptr, IDLE)
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
function reset_hostcall_area!(manager::AreaManager, mod::CuModule)
    hostcall_area = assure_hostcall_area(mod.ctx, required_size(manager))

    kind = kind_config(manager, hostcall_area)
    kind_global = CuGlobal{KindConfig}(mod, KINDCONFIG)
    kind_global[] = kind
end


"""
    check_area(manager::AreaManager, index::Int)::Vector{Ptr{Int64}}

Checks area `index` for open hostmethod calls.
There might be multiple open hostmethod calls related to as certain `index` (1-indexed).
"""
function check_area(manager::AreaManager, ctx::CuContext, index::Int64)::Union{Nothing, Tuple{Int, Vector{Ptr{Int64}}}}
    ptr = convert(Ptr{Int64}, hostcall_areas[ctx])
    ptr += stride(manager) * (index - 1)

    if unsafe_load(ptr) == HOST_CALL
        unsafe_store!(ptr, HOST_HANDLING)
        hostcall = volatile_load(ptr + 8)
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
areas_in(::SimpleAreaManager, ptr::Ptr{Int64}) = Ptr{Int64}[ptr+16]
function areas_in(manager::WarpAreaManager, ptr::Ptr{Int64})
    # TODO
    out = Ptr{Int64}[]

    st = manager.area_size
    for _ in [1:manager.warp_size]
        push!(out, ptr)
        ptr += st
    end

    println("Area count $(length(out))")

    return out
end


"""
    finish_area(manager::AreaManager, index::Int)

Finishes checking an area, notifying the device the host is done handling this function
"""
function finish_area(manager::T, ctx::CuContext, index::Int64) where {T<:AreaManager}
    ptr = convert(Ptr{Int64}, hostcall_areas[ctx])
    ptr += stride(manager) * (index - 1)

    unsafe_store!(ptr+8, 0)
end


"""
    area_count(manager::AreaManager)::Int64

Method returning how many area's should be checked periodically
"""
area_count(manager::SimpleAreaManager) = manager.area_count
area_count(manager::WarpAreaManager) = manager.warp_area_count
