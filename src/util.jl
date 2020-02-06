export VersionRange, @enum_without_prefix

## version range

struct VersionRange
    lower::VersionNumber
    upper::VersionNumber
end

Base.in(v::VersionNumber, r::VersionRange) = (v >= r.lower && v <= r.upper)

import Base.(:)
(:)(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)

Base.intersect(v::VersionNumber, r::VersionRange) =
    v < r.lower ? (r.lower:v) :
    v > r.upper ? (v:r.upper) : (v:v)


## redeclare enum values without a prefix

# this is useful when enum values from an underlying C library, typically prefixed for the
# lack of namespacing in C, are to be used in Julia where we do have module namespacing.
macro enum_without_prefix(enum, prefix)
    if isa(enum, Symbol)
        mod = __module__
    elseif Meta.isexpr(enum, :(.))
        mod = getfield(__module__, enum.args[1])
        enum = enum.args[2].value
    else
        error("Do not know how to refer to $enum")
    end
    enum = getfield(mod, enum)
    prefix = String(prefix)

    ex = quote end
    for instance in instances(enum)
        name = String(Symbol(instance))
        @assert startswith(name, prefix)
        push!(ex.args, :(const $(Symbol(name[length(prefix)+1:end])) = $(mod).$(Symbol(name))))
    end

    return esc(ex)
end
