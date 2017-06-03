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
    if !haskey(ENV, "ONLY_LOAD")
        # check validity of CUDA library
        @debug("Checking validity of $(libcuda_path)")
        if version() != libcuda_version
            error("CUDA library version has changed. Please re-run Pkg.build(\"CUDAdrv\") and restart Julia.")
        end
    end

    __init_logging__()

    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        warn("Running under rr, which is incompatible with CUDA; disabling initialization.")
    elseif !haskey(ENV, "ONLY_LOAD")
        init()
    end
end