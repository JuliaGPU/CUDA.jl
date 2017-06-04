using CUDAnative, CUDAdrv
using Base.Test

# NOTE: all kernel function definitions are prefixed with @eval to force toplevel definition,
#       avoiding boxing as seen in https://github.com/JuliaLang/julia/issues/18077#issuecomment-255215304

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        let fname = tempname()
            try
                ret = nothing
                open(fname, "w") do fout
                    redirect_stdout(fout) do
                        ret = $(esc(ex))
                    end
                end
                ret, readstring(fname)
            finally
                rm(fname, force=true)
            end
        end
    end
end

# Run some code on-device, returning captured standard output
macro on_device(exprs)
    @gensym kernel_fn
    quote
        let
            @eval function $kernel_fn()
                $(esc(exprs))

                return nothing
            end

            @cuda (1,1) $kernel_fn()
            synchronize()
        end
    end
end

# helper function for sinking a value to prevent the callee from getting optimized away
@inline sink(i::Int32) =
    Base.llvmcall("""%slot = alloca i32
                     store volatile i32 %0, i32* %slot
                     %value = load volatile i32, i32* %slot
                     ret i32 %value""", Int32, Tuple{Int32}, i)
@inline sink(i::Int64) =
    Base.llvmcall("""%slot = alloca i64
                     store volatile i64 %0, i64* %slot
                     %value = load volatile i64, i64* %slot
                     ret i64 %value""", Int64, Tuple{Int64}, i)

@testset "CUDAnative" begin

@test length(devices()) > 0
if length(devices()) > 0
    include("base.jl")
    include("codegen.jl")

    # pick most recent device (based on compute capability)
    global dev = nothing
    for newdev in devices()
        if dev == nothing || capability(newdev) > capability(dev)
            dev = newdev
        end
    end
    info("Testing using device $(name(dev))")
    global ctx = CuContext(dev, CUDAdrv.SCHED_BLOCKING_SYNC)

    if capability(dev) < v"2.0"
        warn("native execution not supported on SM < 2.0")
    else
        include("execution.jl")
        include("array.jl")
        include("intrinsics.jl")

        include("examples.jl")
    end
end

end
