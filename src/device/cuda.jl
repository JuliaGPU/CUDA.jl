# wrappers for functionality provided by the CUDA toolkit

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

# functionality from libdevice
#
# > The libdevice library is a collection of NVVM bitcode functions that implement common
# > functions for NVIDIA GPU devices, including math primitives and bit-manipulation
# > functions. These functions are optimized for particular GPU architectures, and are
# > intended to be linked with an NVVM IR module during compilation to PTX.
include(joinpath("cuda", "math.jl"))
# TODO: native mathematical functions, CUDA C programming guide" > "C language extensions"
#       https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
#       see /path/to/cuda/include/sm_20_intrinsics.h

# functionality from libcudadevrt
#
# The libcudadevrt library is a collection of PTX bitcode functions that implement parts of
# the CUDA API for execution on the device, such as device synchronization primitives,
# dynamic kernel APIs, etc.
const cudaError_t = Cint
include(joinpath("cuda", "cooperative_groups.jl"))
include(joinpath("cuda", "dynamic_parallelism.jl"))
