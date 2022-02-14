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
@device_override @noinline Base.throw_boundserror(A, I) =
    @print_and_throw "Out-of-bounds array access"

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @print_and_throw "sincos(x) is only defined for finite x."

# multidimensional.jl
@static if VERSION >= v"1.7-"
    # XXX: the boundscheck change in JuliaLang/julia#42119 has exposed additional issues
    #      with bad code generation by ptxas on <sm_70, as reported with NVIDIA in #3382020.
    @device_override Base.@propagate_inbounds function Base.getindex(iter::CartesianIndices{N,R},
                                                                     I::Vararg{Int, N}) where {N,R}
        if compute_capability() < sv"7"
            CartesianIndex(getindex.(iter.indices, I))
        else
            @boundscheck checkbounds(iter, I...)
            index = map(iter.indices, I) do r, i
                @inbounds getindex(r, i)
            end
            CartesianIndex(index)
        end
    end
end

# range.jl
@static if VERSION >= v"1.7-"
    @device_override quote
            function Base.StepRangeLen{T,R,S,L}(ref::R, step::S, len::Integer,
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
else
    @device_override quote
        function Base.StepRangeLen{T,R,S}(ref::R, step::S, len::Integer,
                                          offset::Integer=1) where {T,R,S}
            if T <: Integer && !isinteger(ref + step)
                @print_and_throw("StepRangeLen{<:Integer} cannot have non-integer step")
            end
            len >= 0 || @print_and_throw("StepRangeLen length cannot be negative")
            1 <= offset <= max(1,len) || @print_and_throw("StepRangeLen: offset must be in [1,...]")
            new(ref, step, len, offset)
        end
    end
end

# LinearAlgebra
@static if VERSION >= v"1.8-"
    @device_override function Base.setindex!(D::LinearAlgebra.Diagonal, v, i::Int, j::Int)
        @boundscheck checkbounds(D, i, j)
        if i == j
            @inbounds D.diag[i] = v
        elseif !iszero(v)
            @print_and_throw("cannot set off-diagonal entry to a nonzero value")
        end
        return v
    end
end

# fastmath.jl
@static if VERSION <= v"1.7-"
## prevent fallbacks to libm
for f in (:acosh, :asinh, :atanh, :cbrt, :cosh, :exp2, :expm1, :log1p, :sinh, :tanh)
    f_fast = Base.FastMath.fast_op[f]
    @eval begin
        @device_override Base.FastMath.$f_fast(x::Float32) = $f(x)
        @device_override Base.FastMath.$f_fast(x::Float64) = $f(x)
    end
end
end
