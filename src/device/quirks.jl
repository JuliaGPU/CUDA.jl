macro print_and_throw(args...)
    quote
        @cuprintln "ERROR: " $(args...) "."
        throw(nothing)
    end
end

# math.jl
@device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @print_and_throw "This operation requires a complex input to return a complex result"
@device_override @noinline Base.Math.throw_exp_domainerror(x) =
    @print_and_throw "Exponentiation yielding a complex result requires a complex argument"

# intfuncs.jl
@device_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::Integer, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::AbstractMatrix, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@device_override @noinline Base.__throw_gcd_overflow(a, b) =
    @print_and_throw "gcd overflow"

# checked.jl
@device_override @noinline Base.Checked.throw_overflowerr_binaryop(op, x, y) =
    @print_and_throw "Binary operation overflowed"
@device_override @noinline Base.Checked.throw_overflowerr_negation(op, x, y) =
    @print_and_throw "Negation overflowed"
@device_override function Base.Checked.checked_abs(x::Base.Checked.SignedInt)
    r = ifelse(x<0, -x, x)
    r<0 && @print_and_throw("checked arithmetic: cannot compute |x|")
    r
end

# boot.jl
@device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @print_and_throw "Inexact conversion"

# abstractarray.jl
@noinline throw_boundserror() =
    @print_and_throw "Out-of-bounds array access"
@device_override @inline Base.throw_boundserror(A, I) = throw_boundserror()

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @print_and_throw "sincos(x) is only defined for finite x."

# range.jl
@eval begin
    @device_override function Base.StepRangeLen{T,R,S,L}(ref::R, step::S, len::Integer,
                                                         offset::Integer=1) where {T,R,S,L}
        if T <: Integer && !isinteger(ref + step)
            @print_and_throw("StepRangeLen{<:Integer} cannot have non-integer step")
        end
        len = convert(L, len)
        len >= zero(len) || @print_and_throw("StepRangeLen length cannot be negative")
        offset = convert(L, offset)
        L1 = oneunit(typeof(len))
        L1 <= offset <= max(L1, len) || @print_and_throw("StepRangeLen: offset must be in [1,...]")
        $(
            Expr(:new, :(StepRangeLen{T,R,S,L}), :ref, :step, :len, :offset)
        )
    end
end

# LinearAlgebra
@device_override function Base.setindex!(D::LinearAlgebra.Diagonal, v, i::Int, j::Int)
    @boundscheck checkbounds(D, i, j)
    if i == j
        @inbounds D.diag[i] = v
    elseif !iszero(v)
        @print_and_throw("cannot set off-diagonal entry to a nonzero value")
    end
    return v
end
