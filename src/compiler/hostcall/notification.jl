
abstract type NotificationPolicy end
required_size(::NotificationPolicy) = 0
kind(::NotificationPolicy) = -1
reset_policy_area!(inner::NotificationPolicy, policy_buffer::Mem.HostBuffer) = () # TODO


"""
    SimpleNotificationPolicy

Hostside policy config struct
"""
mutable struct SimpleNotificationPolicy <: NotificationPolicy
    count::Int64
    indices::Vector{Int64}
end
SimpleNotificationPolicy(count::Int64) = SimpleNotificationPolicy(count, [0 for i in 1:count])
function check_notification(policy::SimpleNotificationPolicy, policy_area::Ptr{Int64}, index::Int64)::Vector{Int64}
    index > policy.count && (error("What are you doing, indexing me so high"); return Int64[])

    out = Int64[]

    if policy.indices[index] < unsafe_load(policy_area, index) # We found a gucci
        policy.indices[index] += 1
        push!(out, index)
    end

    return out
end

"""
    SimpleConfig

Device side policy, passed via const memory
"""
struct SimpleConfig end

kind(::SimpleConfig) = 1
kind(::SimpleNotificationPolicy) = 1

config(::SimpleNotificationPolicy) = SimpleConfig()
required_size(policy::SimpleNotificationPolicy) = policy.count * sizeof(Int64)

function notify_host(::SimpleConfig, policy_ptr::Core.LLVMPtr{Int64,AS.Global}, index::Int32)
    atomic_add!(policy_ptr + sizeof(Int64) * index, 1)
end

function reset_policy_area!(simple::SimpleNotificationPolicy, policy_buffer::Ptr{Int64})
    for i in 1:simple.count
        unsafe_store!(policy_buffer, 0, i)
    end
end


struct NotificationConfig
    kind::Int64
    policy_ptr::Core.LLVMPtr{Int64,AS.Global}
    inner::Union{SimpleConfig}
end

NotificationConfig(inner::NotificationPolicy, policy_buffer::Mem.HostBuffer) = NotificationConfig(kind(inner), reinterpret(Core.LLVMPtr{Int64,AS.Global}, policy_buffer.ptr), config(inner))
function notify_host(config::NotificationConfig, index::Int32)
    notify_host(config.inner, config.policy_ptr, index)
end
