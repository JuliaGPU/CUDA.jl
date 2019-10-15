# Initialization

"""
Initialize the CUDA driver API.

This function is automatically called upon loading the package. You should not have to call
this manually.
"""
function init(flags::Int=0)
    cuInit(flags)
end
