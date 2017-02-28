using CUDAdrv
using Base.Test

using Compat

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

@testset "CUDAdrv" begin

@test devcount() > 0
if devcount() > 0
    include("core.jl")

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

    include("wrappers.jl")
    include("gc.jl")

    include("examples.jl")
end
end