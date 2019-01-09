# CUDA extensions to the C language

# TODO: "CUDA C programming guide" > "C language extensions" lists mathematical functions,
#       without mentioning libdevice. Is this implied, by NVCC always using libdevice,
#       or are there some natively-supported math functions as well?

# yes: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
# see /home/tbesard/CUDA/toolkit/current/include/sm_20_intrinsics.h

include(joinpath("cuda_intrinsics", "memory_shared.jl"))
include(joinpath("cuda_intrinsics", "indexing.jl"))
include(joinpath("cuda_intrinsics", "synchronization.jl"))
include(joinpath("cuda_intrinsics", "warp_vote.jl"))
include(joinpath("cuda_intrinsics", "warp_shuffle.jl"))
include(joinpath("cuda_intrinsics", "output.jl"))
include(joinpath("cuda_intrinsics", "assertion.jl"))
include(joinpath("cuda_intrinsics", "memory_dynamic.jl"))
