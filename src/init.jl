# Initialization

"""
Initialize the CUDA driver API.

This function is automatically called upon loading the package. You should not have to call
this manually.
"""
function init(flags::Int=0)
    @apicall(:cuInit, (Cint,), flags)
end

function __init__()
    configured || return

    if !ispath(libcuda) || version() != libcuda_version
        error("CUDA driver library has changed. Please run Pkg.build(\"CUDAdrv\") and restart Julia.")
    end

    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        @warn "Running under rr, which is incompatible with CUDA; disabling initialization."
    else
        init()
    end
end