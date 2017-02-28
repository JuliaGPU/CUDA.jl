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
                $exprs

                return nothing
            end

            @cuda (1,1) $kernel_fn()
            synchronize()
        end
    end
end

@testset "CUDAnative" begin

@test devcount() > 0
if devcount() > 0
    include("base.jl")
    include("codegen.jl")

    global dev
    dev = CuDevice(0)
    if capability(dev) < v"2.0"
        warn("native execution not supported on SM < 2.0")
    else
        ctx = CuContext(dev, CUDAdrv.SCHED_BLOCKING_SYNC)

        include("execution.jl")
        include("array.jl")
        include("intrinsics.jl")

        include("examples.jl")
    end
end

end
