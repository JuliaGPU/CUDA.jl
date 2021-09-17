macro print_and_throw(args...)
    quote
        @cuprintln "ERROR: " $(args...) "."
        throw(nothing)
    end
end

# math.jl
@device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @print_and_throw "This operation requires a complex input to return a complex result"
@device_override @noinline Base.Math.throw_exp_domainerror(f::Symbol, x) =
    @print_and_throw "Exponentiation yielding a complex result requires a complex argument"

# intfuncs.jl
@device_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::Integer, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::AbstractMatrix, p) =
    @print_and_throw "Cannot raise an integer to a negative power"

# checked.jl
@device_override @noinline Base.Checked.throw_overflowerr_binaryop(op, x, y) =
    @print_and_throw "Binary operation overflowed"
@device_override @noinline Base.Checked.throw_overflowerr_negation(op, x, y) =
    @print_and_throw "Negation overflowed"

# boot.jl
@device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @print_and_throw "Inexact conversion"

# abstractarray.jl
@device_override @noinline Base.throw_boundserror(A, I) =
    @print_and_throw "Out-of-bounds array access"

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @print_and_throw "sincos(x) is only defined for finite x."

# multidimensional.jl
if VERSION >= v"1.7-"
    # XXX: the boundscheck change in JuliaLang/julia#42119 has exposed additional issues
    #      with bad code generation by ptxas, so revert that changen for now.
    @device_override Base.@propagate_inbounds function Base.getindex(iter::CartesianIndices{N,R},
                                                                     I::Vararg{Int, N}) where {N,R}
        CartesianIndex(getindex.(iter.indices, I))
    end
end
