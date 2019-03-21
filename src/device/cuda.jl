# wrappers for functionality provided by the CUDA toolkit

include(joinpath("cuda", "libdevice.jl"))
include(joinpath("cuda", "libcudadevrt.jl"))

# extensions to the C language
include(joinpath("cuda", "memory_shared.jl"))
include(joinpath("cuda", "indexing.jl"))
include(joinpath("cuda", "synchronization.jl"))
include(joinpath("cuda", "warp_vote.jl"))
include(joinpath("cuda", "warp_shuffle.jl"))
include(joinpath("cuda", "output.jl"))
include(joinpath("cuda", "assertion.jl"))
include(joinpath("cuda", "memory_dynamic.jl"))
include(joinpath("cuda", "misc.jl"))

# TODO: "CUDA C programming guide" > "C language extensions" lists mathematical functions,
#       without mentioning libdevice. Is this implied, by NVCC always using libdevice,
#       or are there some natively-supported math functions as well?
#
#       yes: see /path/to/cuda/include/sm_20_intrinsics.h
#       https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
