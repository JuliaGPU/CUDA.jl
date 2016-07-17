# Determine which type to pre-convert objects to for use on a CUDA device.
#
# The resulting object type will be used as a starting point to determine the final
# specialization and argument types (there might be other conversions, eg. factoring in the
# ABI). This is different from `cconvert` in that we don't know which type to convert to.
cudaconvert{T}(::Type{T}) = T
