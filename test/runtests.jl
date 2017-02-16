using CUDAdrv
using Base.Test

using Compat

@test devcount() > 0

# NOTE: based on test/pkg.jl::grab_outputs, only grabs STDOUT without capturing exceptions
macro grab_output(ex)
    quote
        OLD_STDOUT = STDOUT

        foutname = tempname()
        fout = open(foutname, "w")

        local ret
        local caught_ex = nothing
        try
            redirect_stdout(fout)
            ret = $(esc(ex))
        catch ex
            caught_ex = nothing
        finally
            redirect_stdout(OLD_STDOUT)
            close(fout)
        end
        out = readstring(foutname)
        rm(foutname)
        if caught_ex != nothing
            throw(caught_ex)
        end

        ret, out
    end
end

include("core.jl")
