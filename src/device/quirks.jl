# Overlay quirks replacing GPU-incompatible methods

# most of the quirks below deal with exceptions, avoiding calls to often purposefully
# underspecialized constructors that would result in allocations and/or dynamic dispatches.
# GPUCompiler.jl cannot detect the type of these exceptions (only those thrown by C code,
# as identified by calls to specific functions like `throw_boundserror`), so store extra
# information about the exception type and reason in the `ExceptionInfo` struct, where it
# will be read by the GPU runtime and printed to the console.

macro gputhrow(subtype, reason)
    quote
        info = kernel_state().exception_info
        info.subtype = @strptr $subtype
        info.reason = @strptr $reason
        throw(nothing)
    end
end

# math.jl
@device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @gputhrow "DomainError" "This operation requires a complex input to return a complex result"
@device_override @noinline Base.Math.throw_exp_domainerror(x) =
    @gputhrow "DomainError" "Exponentiation yielding a complex result requires a complex argument"

# intfuncs.jl
@device_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::Integer, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::AbstractMatrix, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"
@device_override @noinline Base.__throw_gcd_overflow(a, b) =
    @gputhrow "OverflowError" "gcd overflow"

# checked.jl
@device_override @noinline Base.Checked.throw_overflowerr_binaryop(op, x, y) =
    @gputhrow "OverflowError" "Binary operation overflowed"
@device_override @noinline Base.Checked.throw_overflowerr_negation(op, x, y) =
    @gputhrow "OverflowError" "Negation overflowed"
@device_override function Base.Checked.checked_abs(x::Base.Checked.SignedInt)
    r = ifelse(x<0, -x, x)
    r<0 && @gputhrow("OverflowError", "checked arithmetic: cannot compute |x|")
    r
end

# boot.jl
@device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @gputhrow "InexactError" "Inexact conversion"

# abstractarray.jl
@noinline throw_boundserror() =
    @gputhrow "BoundsError" "Out-of-bounds array access"
@device_override @inline Base.throw_boundserror(A, I) = throw_boundserror()

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @gputhrow "DomainError" "sincos(x) is only defined for finite x."

# range.jl
@eval begin
    @device_override function Base.StepRangeLen{T,R,S,L}(ref::R, step::S, len::Integer,
                                                         offset::Integer=1) where {T,R,S,L}
        if T <: Integer && !isinteger(ref + step)
            @gputhrow("ArgumentError", "StepRangeLen{<:Integer} cannot have non-integer step")
        end
        len = convert(L, len)
        len >= zero(len) || @gputhrow("ArgumentError", "StepRangeLen length cannot be negative")
        offset = convert(L, offset)
        L1 = oneunit(typeof(len))
        L1 <= offset <= max(L1, len) || @gputhrow("ArgumentError", "StepRangeLen: offset must be in [1,...]")
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
        @gputhrow("ArgumentError", "cannot set off-diagonal entry to a nonzero value")
    end
    return v
end

# rational.jl
@device_override @noinline Base.__throw_rational_argerror_zero(::Type{T}) where {T} =
    @gputhrow "ArgumentError" "invalid rational: 0//0"
@static if VERSION < v"1.11.0-DEV.708"
@device_override @noinline Base.__throw_rational_argerror_typemin(::Type{T}) where {T} =
    @gputhrow "OverflowError" "rational numerator is typemin"
end
