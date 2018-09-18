# common functionality

struct CompilerContext
    # core invocation
    f::Core.Function
    tt::DataType
    cap::VersionNumber
    kernel::Bool

    # optional properties
    alias::Union{Nothing,String}
    minthreads::Union{Nothing,CuDim}
    maxthreads::Union{Nothing,CuDim}
    blocks_per_sm::Union{Nothing,Integer}
    maxregs::Union{Nothing,Integer}

    CompilerContext(f, tt, cap, kernel; alias=nothing,
                    minthreads=nothing, maxthreads=nothing, blocks_per_sm=nothing, maxregs=nothing) =
        new(f, tt, cap, kernel, alias, minthreads, maxthreads, blocks_per_sm, maxregs)
end

function signature(ctx::CompilerContext)
    fn = typeof(ctx.f).name.mt.name
    args = join(ctx.tt.parameters, ", ")
    return "$fn($(join(ctx.tt.parameters, ", ")))"
end


abstract type AbstractCompilerError <: Exception end

struct CompilerError <: AbstractCompilerError
    ctx::CompilerContext
    message::String
    bt::StackTraces.StackTrace
    meta::Dict

    CompilerError(ctx::CompilerContext, message="unknown error",
                  bt=StackTraces.StackTrace(); kwargs...) =
        new(ctx, message, bt, kwargs)
end

function Base.showerror(io::IO, err::CompilerError)
    print(io, "CompilerError: could not compile $(signature(err.ctx)); $(err.message)")
    for (key,val) in err.meta
        print(io, "\n- $key = $val")
    end
    Base.show_backtrace(io, err.bt)
end


# maintain our own "global unique" suffix for disambiguating kernels
globalUnique = 0
