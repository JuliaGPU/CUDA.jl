using CUDAnative, CUDAdrv
using Base.Test

# NOTE: all kernel function definitions are prefixed with @eval to force toplevel definition,
#       avoiding boxing as seen in https://github.com/JuliaLang/julia/issues/18077#issuecomment-255215304

# a composite type to test for more complex element types
immutable RGB{T}
    r::T
    g::T
    b::T
end

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

@testset "CUDAnative" begin

@test devcount() > 0
if devcount() > 0
    include("base.jl")
    include("codegen.jl")

    # pick most recent device (based on compute capability)
    global dev = nothing
    for i in 0:devcount()-1
        newdev = CuDevice(i)
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
