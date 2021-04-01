"""
    HostRef

Struct to help return host side only arguments to the device
These can then be used to call other hostmethods

For examples useful when first creating a file then reading from
it on the device.
"""
struct HostRef
    index::Int32
end

const host_refs = Vector{WeakRef}()
const host_refs_lk = ReentrantLock()

Base.show(io::IO, t::HostRef) = print(io, "HostRef to $(host_refs[t.index])")
Base.convert(::Type{HostRef}, t::HostRef) = t

function Base.convert(::Type{HostRef}, t::T) where {T}
    lock(host_refs_lk) do
        push!(host_refs, WeakRef(t))
    end

    return HostRef(length(host_refs))
end

Base.convert(::Type{Any}, t::HostRef) = host_refs[t.index].value
Adapt.adapt(::InvAdaptor, t::HostRef) = host_refs[t.index].value
