# CUDA index specialization policy: one bounded bin and one unbounded bin.
# Unrelated scalar data and array byte sizes do not participate.

@generated function index_ranges(::Type{T}) where {T}
    (T isa DataType && isbitstype(T) && sizeof(T) > 0 && !isprimitivetype(T)) ||
        return :(())
    fields = Expr[]
    for i in 1:fieldcount(T)
        key = T <: Tuple ? i : fieldname(T, i)
        push!(fields, :(prefix_argument_ranges($(QuoteNode(key)),
                                              index_ranges($(fieldtype(T, i))))))
    end
    # Call downstream hooks in the generated body, not the generator's definition world.
    return Expr(:tuple, (Expr(:..., f) for f in fields)...)
end
# Select explicit indexing objects, not arbitrary integer payloads.
index_ranges(::Type{Base.OneTo{Int}}) = ((:stop,) => 0:Int(typemax(Int32)),)
index_ranges(::Type{UnitRange{Int}}) =
    ((:start,) => Int(typemin(Int32)):Int(typemax(Int32)),
     (:stop,) => Int(typemin(Int32)):Int(typemax(Int32)))

index_ranges(::Type{StepRange{Int,Int}}) =
    ((:start,) => Int(typemin(Int32)):Int(typemax(Int32)),
     (:step,) => Int(typemin(Int32)):Int(typemax(Int32)),
     (:stop,) => Int(typemin(Int32)):Int(typemax(Int32)))
function index_ranges(::Type{CartesianIndex{N}}) where {N}
    ranges = ()
    for i in 1:N
        ranges = (ranges..., (:I, i) => Int(typemin(Int32)):Int(typemax(Int32)))
    end
    return ranges
end

@inline function prefix_argument_ranges(key, ranges)
    ranges isa Tuple || throw(ArgumentError("index_ranges must return a tuple"))
    map(ranges) do fact
        fact isa Pair && first(fact) isa Tuple ||
            throw(ArgumentError("argument ranges must be tuple-path => range pairs"))
        (key, first(fact)...) => last(fact)
    end
end

function index_ranges(::Type{<:CuDeviceArray{T,N}}) where {T,N}
    ranges = ()
    for i in 1:N
        ranges = (ranges..., (:dims, i) => 0:typemax(Int32))
    end
    return (ranges..., (:len,) => 0:typemax(Int32))
end

# A device view's cached offset and stride are index-derived values. Leaving them wide
# would lose narrowing for strided views even when their parent and indices fit.
function index_ranges(::Type{S}) where {T,N,P<:CuDeviceArray,S<:SubArray{T,N,P}}
    ranges = ()
    for field in (:parent, :indices)
        for (path, range) in index_ranges(fieldtype(S, field))
            ranges = (ranges..., (field, path...) => range)
        end
    end
    return (ranges..., (:offset1,) => typemin(Int32):typemax(Int32),
                       (:stride1,) => typemin(Int32):typemax(Int32))
end

function fieldtype_at_path(T, path)
    for key in path
        T = fieldtype(T, key)
    end
    T
end

@inline @generated function field_at_path(x, path::Tuple{Vararg{Any,N}}) where {N}
    ex = :x
    for i in 1:N
        ex = :(getfield($ex, path[$i]))
    end
    ex
end

@inline function check_argument_range(x, fact)
    path, range = fact
    value = field_at_path(x, path)
    T = typeof(value)
    lo, hi = T(first(range)), T(last(range))
    if T === Bool
        lo <= value <= hi
    else
        # One unsigned comparison, including ranges crossing zero. Avoid length(range).
        U = unsigned(T)
        (value % U - lo % U) <= (hi % U - lo % U)
    end
end

@inline @generated function check_argument_ranges(x, ranges::Tuple{Vararg{Any,N}}) where {N}
    checks = [:(check_argument_range(x, ranges[$i])) for i in 1:N]
    isempty(checks) ? true : foldl((a, b) -> :($a & $b), checks)
end

Base.@assume_effects :foldable static_index_ranges(::Type{T}) where {T} = index_ranges(T)

# Precompute the two bins from types; values select one without allocating metadata.
# This is CUDA's policy. GPUCompiler only sees the selected range facts.
Base.@constprop :aggressive Base.@assume_effects :foldable function candidate_index_ranges(::Type{F}, ::Type{TT}) where {F,TT<:Tuple}
    ranges = map((F, TT.parameters...)) do T
        filter(index_ranges(T)) do (path, range)
            leaf = fieldtype_at_path(T, path)
            first(range) != typemin(leaf) || last(range) != typemax(leaf)
        end
    end
    all(isempty, ranges) ? () : ranges
end

@inline function argument_index_ranges(f::F, args::TT) where {F,TT<:Tuple}
    ranges = candidate_index_ranges(F, TT)
    index_ranges_hold(f, args) ? ranges : ()
end

# Resolve hooks in the generated body, so downstream definitions are visible.
@inline @generated function index_ranges_hold(f::F, args::TT) where {F,TT<:Tuple}
    checks = Expr[]
    for (i, T) in enumerate((F, TT.parameters...))
        value = i == 1 ? :f : :(args[$(i-1)])
        push!(checks, :(check_argument_ranges($value, static_index_ranges($T))))
    end
    isempty(checks) ? true : foldl((a,b) -> :($a && $b), checks)
end

# Dispatch types and value layouts differ for Type{T}. Check the known dispatch
# signature without constructing Type{T} through the runtime type cache.
@inline @generated function specialization_types_match(::Type{TT}, args::AT) where {TT<:Tuple,AT<:Tuple}
    fieldcount(TT) == fieldcount(AT) || return false
    checks = Expr[]
    for i in 1:fieldcount(TT)
        T = fieldtype(TT, i)
        if T isa DataType && T.name === Base.unwrap_unionall(Type).name
            push!(checks, :(args[$i] === $(QuoteNode(T.parameters[1]))))
        else
            push!(checks, :(typeof(args[$i]) === $T))
        end
    end
    isempty(checks) ? true : foldl((a, b) -> :($a && $b), checks)
end

# Retained kernels must still satisfy the selected bin on every launch.
@inline function check_index_ranges(ranges, f::F, ::Type{TT}, args::AT) where {F,TT<:Tuple,AT<:Tuple}
    isempty(ranges) || (specialization_types_match(TT, args) &&
                       ranges === argument_index_ranges(f, args))
end
