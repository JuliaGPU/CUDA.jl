# helpers for writing device functionality

# helper type for writing Int32 literals
# TODO: upstream this
struct Literal{T} end
Base.:(*)(x, ::Type{Literal{T}}) where {T} = T(x)
const i32 = Literal{Int32}

# local method table for device functions
@static if isdefined(Base.Experimental, Symbol("@overlay"))
Base.Experimental.@MethodTable(method_table)
else
const method_table = nothing
end

# list of overrides (only for Julia 1.6)
const overrides = Expr[]

macro device_override(ex)
    if Meta.isexpr(ex, :quote)
        # TODO: support @device_override @eval ...
        ex = Base.eval(__module__, ex)
        # we only support single-expression blocks; strip the line number info
        @assert Meta.isexpr(ex, :block) && length(ex.args) == 2
        ex = ex.args[end]
    end
    ex = macroexpand(__module__, ex)
    if Meta.isexpr(ex, :call)
        @show ex = eval(ex)
        error()
    end
    code = quote
        $GPUCompiler.@override(CUDA.method_table, $ex)
    end
    if isdefined(Base.Experimental, Symbol("@overlay"))
        return esc(code)
    else
        push!(overrides, code)
        return
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
        $(combinedef(def))
        @device_override $ex
    end)
end

macro device_functions(ex)
    ex = macroexpand(__module__, ex)

    # recursively prepend `@device_function` to all function definitions
    function rewrite(block)
        out = Expr(:block)
        for arg in block.args
            if Meta.isexpr(arg, :block)
                # descend in blocks
                push!(out.args, rewrite(arg))
            elseif Meta.isexpr(arg, [:function, :(=)])
                # rewrite function definitions
                push!(out.args, :(@device_function $arg))
            else
                # preserve all the rest
                push!(out.args, arg)
            end
        end
        out
    end

    esc(rewrite(ex))
end
