# common functionality

struct CompilerContext
    # core invocation
    f::Core.Function
    tt::DataType
    cap::VersionNumber
    kernel::Bool

    # optional properties
    minthreads::Union{Nothing,CuDim}
    maxthreads::Union{Nothing,CuDim}
    blocks_per_sm::Union{Nothing,Integer}
    maxregs::Union{Nothing,Integer}

    CompilerContext(f, tt, cap, kernel;
                    minthreads=nothing, maxthreads=nothing, blocks_per_sm=nothing, maxregs=nothing) =
        new(f, tt, cap, kernel, minthreads, maxthreads, blocks_per_sm, maxregs)
end

function signature(ctx::CompilerContext)
    fn = typeof(ctx.f).name.mt.name
    args = join(ctx.tt.parameters, ", ")
    return "$fn($(join(ctx.tt.parameters, ", ")))"
end


struct KernelError <: Exception
    ctx::CompilerContext
    message::String
    help::Union{Nothing,String}
    bt::StackTraces.StackTrace

    KernelError(ctx::CompilerContext, message::String, help=nothing;
                bt=StackTraces.StackTrace()) =
        new(ctx, message, help, bt)
end

function Base.showerror(io::IO, err::KernelError)
    println(io, "GPU compilation of $(signature(err.ctx)) failed")
    println(io, "KernelError: $(err.message)")
    println(io)
    println(io, something(err.help, "Try inspecting the generated code with any of the @device_code_... macros."))
    Base.show_backtrace(io, err.bt)
end


struct InternalCompilerError <: Exception
    ctx::CompilerContext
    message::String
    meta::Dict
    InternalCompilerError(ctx, message; kwargs...) = new(ctx, message, kwargs)
end

function Base.showerror(io::IO, err::InternalCompilerError)
    println(io, """CUDAnative.jl encountered an unexpected internal compiler error.
                   Please file an issue attaching the following information, including the backtrace,
                   as well as a reproducible example (if possible).""")

    println(io, "\nInternalCompilerError: $(err.message)")

    println(io, "\nCompiler invocation:")
    for field in fieldnames(CompilerContext)
        println(io, " - $field = $(repr(getfield(err.ctx, field)))")
    end

    if !isempty(err.meta)
        println(io, "\nAdditional information:")
        for (key,val) in err.meta
            println(io, " - $key = $(repr(val))")
        end
    end

    println(io, "\nInstalled packages:")
    for (pkg,ver) in Pkg.installed()
        println(io, " - $pkg = $ver")
    end

    println(io)
    versioninfo(io)
end

macro compiler_assert(ex, ctx, kwargs...)
    msg = "$ex, at $(__source__.file):$(__source__.line)"
    return :($(esc(ex)) ? $(nothing)
                        : throw(InternalCompilerError($(esc(ctx)), $msg;
                                                      $(map(esc, kwargs)...)))
            )
end


# maintain our own "global unique" suffix for disambiguating kernels
globalUnique = 0
