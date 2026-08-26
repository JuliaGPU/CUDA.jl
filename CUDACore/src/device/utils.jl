# helpers for writing device functionality

# local method table for device functions
@static if isdefined(Base.Experimental, Symbol("@overlay"))
Base.Experimental.@MethodTable(method_table)
else
const method_table = nothing
end

@public @device_override, @device_function, @device_functions

macro device_override(ex)
    ex = macroexpand(__module__, ex)
    if VERSION >= v"1.12.0-DEV.745" || v"1.11-rc1" <= VERSION < v"1.12-"
        # this requires that the overlay method f′ is consistent with f, i.e.,
        #   - if f(x) returns a value, f′(x) must return the identical value.
        #   - if f(x) throws an exception, f′(x) must also throw an exception
        #     (although the exceptions do not need to be identical).
        esc(quote
            Base.Experimental.@consistent_overlay($(CUDACore.method_table), $ex)
        end)
    else
        esc(quote
            Base.Experimental.@overlay($(CUDACore.method_table), $ex)
        end)
    end
end

macro device_function(ex)
    ex = macroexpand(__module__, ex)
    def = splitdef(ex)

    # generate a function that errors
    def[:body] = quote
        error("This function is not intended for use on the CPU")
    end

    esc(quote
        # `@__doc__` makes docstrings apply to the CPU stub (i.e., to the function
        # binding), so that `@device_functions` can rewrite documented definitions
        Base.@__doc__ $(combinedef(def))

        # NOTE: no use of `@consistent_overlay` here because the regular function errors
        Base.Experimental.@overlay($(CUDACore.method_table), $ex)
    end)
end

macro device_functions(ex)
    # recursively prepend `@device_function` to all function definitions.
    #
    # NOTE: definitions are rewritten *before* macro expansion, where possible. In
    #       particular, docstrings (`Core.@doc`) are handled by descending into the
    #       documented expression, because the shape of `@doc`'s expansion has changed
    #       repeatedly (Julia 1.13 wraps the definition in `if true; val = ...; end`),
    #       and any definition that escapes this rewrite ends up in the global method
    #       table as a CPU-callable function containing GPU-only intrinsics, which
    #       breaks ahead-of-time compilation (JuliaGPU/GPUCompiler.jl#611).
    isdef(ex) = Meta.isexpr(ex, :function) ||
                (Meta.isexpr(ex, :(=)) && Meta.isexpr(ex.args[1], (:call, :where, :(::))))
    isdoc(m) = m === GlobalRef(Core, Symbol("@doc")) || m === Symbol("@doc") ||
               (Meta.isexpr(m, :.) && m.args[end] == QuoteNode(Symbol("@doc")))
    function rewrite(ex)
        if isdef(ex)
            return :(@device_function $ex)
        elseif Meta.isexpr(ex, :macrocall) && isdoc(ex.args[1])
            # docstring: rewrite the documented expression
            return Expr(:macrocall, ex.args[1:end-1]..., rewrite(ex.args[end]))
        elseif Meta.isexpr(ex, :macrocall)
            # other macros (e.g. `@inline`): expand, then rewrite the result
            return rewrite(macroexpand(__module__, ex))
        elseif Meta.isexpr(ex, (:block, :if, :elseif))
            # descend into blocks, and into conditionals (the `@doc` expansion on
            # Julia 1.13 wraps definitions in `if true ... end`)
            return Expr(ex.head, map(rewrite, ex.args)...)
        elseif Meta.isexpr(ex, :(=)) && isa(ex.args[1], Symbol)
            # assignment of a definition to a temporary (also from `@doc` expansion)
            return Expr(:(=), ex.args[1], rewrite(ex.args[2]))
        else
            # preserve all the rest
            return ex
        end
    end

    esc(rewrite(ex))
end

## alignment API

# we don't expose this as Aligned{N}, because we want to have the T typevar first
# to facilitate use in function signatures as ::Aligned{<:T}

struct Aligned{T, N}
    data::T
end

alignment(::Aligned{<:Any, N}) where {N} = N
Base.getindex(x::Aligned) = x.data

"""
    CUDA.align{N}(obj)

Construct an aligned object, providing alignment information to APIs that require it.
"""
struct align{N} end
(::Type{align{N}})(data::T) where {T,N} = Aligned{T,N}(data)

# default alignment for common types
Aligned(x::Aligned) = x
Aligned(x::Ptr{T}) where T = align{Base.datatype_alignment(T)}(x)
Aligned(x::LLVMPtr{T}) where T = align{Base.datatype_alignment(T)}(x)
