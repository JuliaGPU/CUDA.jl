
abstract type NotificationPolicy end
required_size(::NotificationPolicy) = 0
kind(::NotificationPolicy) = -1
reset_policy_area!(inner::NotificationPolicy, policy_buffer::Mem.HostBuffer) = () # TODO
poll_slices(::NotificationPolicy, wanted::Int64) = []

struct NoPolicy <: NotificationPolicy end
NoPolicy(_x) = NoPolicy()


# Util function
function line_split(count, wanted)
    if wanted > count
        return [(i, i) for i in 1:count]
    end

    out = []
    current = 0.0
    offset = count / wanted

    for i in 1:wanted
        start = floor(Int, current)
        current = current + offset
        push!(out, (start + 1, floor(Int, current)))
    end

    return out
end


"""
    SimpleNotificationPolicy

Hostside policy config struct
"""
mutable struct SimpleNotificationPolicy <: NotificationPolicy
    count::Int64
    indices::Vector{Int64}
end
SimpleNotificationPolicy(count::Int64) = SimpleNotificationPolicy(count, [0 for i in 1:count])
poll_slices(simple::SimpleNotificationPolicy, wanted::Int64) = line_split(simple.count, wanted)

function Base.show(io::IO, policy::SimpleNotificationPolicy)
    print(io, "SimpleNotificationPolicy($(policy.count))")
end



required_size(policy::SimpleNotificationPolicy) = policy.count * sizeof(Int64)


function check_notification(policy::SimpleNotificationPolicy, policy_area::Ptr{Int64}, index::Int64)::Vector{Int64}
    index > policy.count && (error("What are you doing, indexing me so high"); return Int64[])

    out = Int64[]

    if policy.indices[index] < unsafe_load(policy_area, index) # We found a gucci
        policy.indices[index] += 1
        push!(out, index)
    end

    return out
end


function reset_policy_area!(simple::SimpleNotificationPolicy, policy_buffer::Ptr{Int64})
    for i in 1:simple.count
        unsafe_store!(policy_buffer, 0, i)
    end
end


"""
    SimpleConfig

Device side policy, passed via const memory
"""
struct SimpleConfig end
kind(::SimpleConfig) = 1
kind(::SimpleNotificationPolicy) = 1

config(::SimpleNotificationPolicy) = SimpleConfig()

function notify_host(::SimpleConfig, policy_ptr::Core.LLVMPtr{Int64,AS.Global}, index::Int32)
    atomic_add!(policy_ptr + sizeof(Int64) * index, 1)
end


mutable struct TreeNotificationPolicy{Depth} <: NotificationPolicy
    indices::Vector{Vector{Int64}}
    tree_size::Int64
    areas_per_tree::Int64
    tree_count::Int64
end

function Base.show(io::IO, policy::TreeNotificationPolicy{Depth}) where {Depth}
    print(io, "SimpleNotificationPolicy($(length(policy.indices)), depth=$(Depth))")
end


function TreeNotificationPolicy{Depth}(count::Int64) where {Depth}
    areas_per_tree = 1 << (Depth - 1)
    many_trees = div(count, areas_per_tree, RoundUp)

    nodes_per_tree = 1 << Depth - 1

    # binary double indexed tree (in array)
    TreeNotificationPolicy{Depth}([[0 for i in 1:nodes_per_tree] for j in 1:many_trees], nodes_per_tree, areas_per_tree, many_trees)
end

required_size(policy::TreeNotificationPolicy) = policy.tree_size * policy.tree_count * sizeof(Int64)
poll_slices(tree::TreeNotificationPolicy, wanted::Int64) = line_split(tree.tree_count, wanted)


# one based index
function check_notification(policy::TreeNotificationPolicy{Depth}, policy_area::Ptr{Int64}, index::Int64)::Vector{Int64} where {Depth}
    children(x) = (x*2, x*2+1)
    out = Int64[]

    current_tree = policy.indices[index]
    offset = policy.tree_size * (index-1)

    current_tree_area_offset = (index - 1) * policy.areas_per_tree

    # check
    if current_tree[1] < unsafe_load(policy_area, offset+1)
        current_tree[1] = unsafe_load(policy_area, offset+1)

        work_list = Int64[1]
        # At least one thingy
        for i in 2:Depth
            work_list = [x for y in work_list for x in children(y) if current_tree[x] < unsafe_load(policy_area, offset + x)]

            for j in work_list
                current_tree[j] = unsafe_load(policy_area, offset + j)
            end
        end

        b = (1 << (Depth - 1)) - 1
        out = [current_tree_area_offset + x - b for x in work_list]
    end


    return out
end


function reset_policy_area!(policy::TreeNotificationPolicy, policy_buffer::Ptr{Int64})
    # ptr    = convert(CuPtr{Int64}, policy_buffer)
    # cuarray = unsafe_wrap(CuArray{Int64}, ptr, policy.tree_size * policy.tree_count)
    # fill!(cuarray, 0)
    for i in 1:(policy.tree_size * policy.tree_count)
        unsafe_store!(policy_buffer, 0, i)
    end


end

function reset_policy_area!(policy::TreeNotificationPolicy, policy_buffer::Mem.HostBuffer)
    ptr    = convert(CuPtr{Int64}, policy_buffer)
    ptr_u = reinterpret(CuPtr{UInt32}, ptr)
    Mem.set!(ptr_u, 0, policy.tree_size * policy.tree_count * 2)

    # cuarray = unsafe_wrap(CuArray{Int64}, ptr, policy.tree_size * policy.tree_count)
    # fill!(cuarray, 0)
    # for i in 1:(policy.tree_size * policy.tree_count)
    #     unsafe_store!(policy_buffer, 0, i)
    # end


end


struct TreeConfig
    height::Int64
    tree_size::Int64
    areas_per_tree::Int64
    tree_count::Int64
end

kind(::TreeConfig) = 2
kind(::TreeNotificationPolicy) = 2

config(tree::TreeNotificationPolicy{H}) where {H} = TreeConfig(H, tree.tree_size, tree.areas_per_tree, tree.tree_count)

# zero based index
function notify_host(conf::TreeConfig, policy_ptr::Core.LLVMPtr{Int64,AS.Global}, index::Int32)
    threadfence()
    index_in_tree = (index % conf.areas_per_tree) + (1 << (conf.height - 1)) - 1
    offset = conf.tree_size * div(index, conf.areas_per_tree)

    zeros = 0
    for i in 1:conf.height
        atomic_add!(policy_ptr + sizeof(Int64) * (offset + index_in_tree), 1)

        if index_in_tree == 0
            zeros += 1
        end
        index_in_tree = div(index_in_tree - 1, 2)
    end

    zeros == 1 || @cuprintln("you done goofed here tree $zeros")

    index_in_tree == 0 || @cuprintln("you done goofed here too")
end


struct NotificationConfig
    kind::Int64
    policy_ptr::Core.LLVMPtr{Int64,AS.Global}
    inner::Union{SimpleConfig, TreeConfig}
end

NotificationConfig(inner::NotificationPolicy, policy_buffer::Mem.HostBuffer) = NotificationConfig(kind(inner), reinterpret(Core.LLVMPtr{Int64,AS.Global}, policy_buffer.ptr), config(inner))
function notify_host(config::NotificationConfig, index::Int32)
    notify_host(config.inner, config.policy_ptr, index)
end
