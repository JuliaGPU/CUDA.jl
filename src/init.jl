# Initialization

"""
Initialize the CUDA driver API.
"""
function init(flags=0)
    @apicall(:cuInit, (Cint,), flags)
end

function __init__()
    # check validity of CUDA library
    @debug("Checking validity of $(libcuda_path)")
    if version() != libcuda_version
        error("CUDA library version has changed. Please re-run Pkg.build(\"CUDAdrv\") and restart Julia.")
    end

    __init_logging__()
    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        warn("Running under rr, which is incompatible with CUDA; disabling initialization.")
    else
        init()
    end
end