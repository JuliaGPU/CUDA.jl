# CUDA extensions to the C language

# TODO: "CUDA C programming guide" > "C language extensions" lists mathematical functions,
#       without mentioning libdevice. Is this implied, by NVCC always using libdevice,
#       or are there some natively-supported math functions as well?

# yes: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
# see /home/tbesard/CUDA/toolkit/current/include/sm_20_intrinsics.h

include(joinpath("intrinsics", "memory_shared.jl"))
include(joinpath("intrinsics", "indexing.jl"))
include(joinpath("intrinsics", "synchronization.jl"))
include(joinpath("intrinsics", "warp_vote.jl"))
include(joinpath("intrinsics", "warp_shuffle.jl"))
include(joinpath("intrinsics", "output.jl"))
include(joinpath("intrinsics", "assertion.jl"))
