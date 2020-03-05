# common functionality

Base.@kwdef struct CompilerJob
    # core invocation
    f::Base.Callable
    tt::DataType
    cap::VersionNumber
    kernel::Bool

    # optional properties
    minthreads::Union{Nothing,CuDim} = nothing
    maxthreads::Union{Nothing,CuDim} = nothing
    blocks_per_sm::Union{Nothing,Integer} = nothing
    maxregs::Union{Nothing,Integer} = nothing
    name::Union{Nothing,String} = nothing
end

CompilerJob(f, tt, cap, kernel; kwargs...) =
    CompilerJob(f=f, tt=tt, cap=cap, kernel=kernel; kwargs...)

function Base.show(io::IO, job::CompilerJob)
    print(io, "CUDAnative.CompilerJob for $(job.f)($(join(job.tt.parameters, ", ")))")

    print(io, " (cap=$(job.cap.major).$(job.cap.minor)")
    job.kernel && print(io, ", kernel=true")
    job.minthreads !== nothing && print(io, ", minthreads=$(job.minthreads)")
    job.maxthreads !== nothing && print(io, ", maxthreads=$(job.maxthreads)")
    job.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.blocks_per_sm)")
    job.maxregs !== nothing && print(io, ", maxregs=$(job.maxregs)")
    job.name !== nothing && print(io, ", name=$(job.name)")
    print(io, ")")
end

# global job reference
# FIXME: thread through `job` everywhere (deadlocks the Julia compiler when doing so with
#        the LLVM passes in CUDAnative)
current_job = nothing


function signature(job::CompilerJob)
    fn = something(job.name, nameof(job.f))
    args = join(job.tt.parameters, ", ")
    return "$fn($(join(job.tt.parameters, ", ")))"
end


struct KernelError <: Exception
    job::CompilerJob
    message::String
    help::Union{Nothing,String}
    bt::StackTraces.StackTrace

    KernelError(job::CompilerJob, message::String, help=nothing;
                bt=StackTraces.StackTrace()) =
        new(job, message, help, bt)
end

function Base.showerror(io::IO, err::KernelError)
    println(io, "GPU compilation of $(signature(err.job)) failed")
    println(io, "KernelError: $(err.message)")
    println(io)
    println(io, something(err.help, "Try inspecting the generated code with any of the @device_code_... macros."))
    Base.show_backtrace(io, err.bt)
end


struct InternalCompilerError <: Exception
    job::CompilerJob
    message::String
    meta::Dict
    InternalCompilerError(job, message; kwargs...) = new(job, message, kwargs)
end

function Base.showerror(io::IO, err::InternalCompilerError)
    println(io, """CUDAnative.jl encountered an unexpected internal compiler error.
                   Please file an issue attaching the following information, including the backtrace,
                   as well as a reproducible example (if possible).""")

    println(io, "\nInternalCompilerError: $(err.message)")

    println(io, "\nCompiler invocation:")
    for field in fieldnames(CompilerJob)
        println(io, " - $field = $(repr(getfield(err.job, field)))")
    end

    if !isempty(err.meta)
        println(io, "\nAdditional information:")
        for (key,val) in err.meta
            println(io, " - $key = $(repr(val))")
        end
    end

    let Pkg = Base.require(Base.PkgId(Base.UUID((0x44cfe95a1eb252ea, 0xb672e2afdf69b78f)), "Pkg"))
        println(io, "\nInstalled packages:")
        for (pkg,ver) in Pkg.installed()
            println(io, " - $pkg = $(repr(ver))")
        end
    end

    println(io)
    versioninfo(io)
end

macro compiler_assert(ex, job, kwargs...)
    msg = "$ex, at $(__source__.file):$(__source__.line)"
    return :($(esc(ex)) ? $(nothing)
                        : throw(InternalCompilerError($(esc(job)), $msg;
                                                      $(map(esc, kwargs)...)))
            )
end
