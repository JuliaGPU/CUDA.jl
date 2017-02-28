using CUDAdrv
using Base.Test

using Compat

@test devcount() > 0

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
include("core.jl")
include("examples.jl")
end