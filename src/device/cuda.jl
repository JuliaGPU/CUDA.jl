# wrappers for functionality provided by the CUDA toolkit

# extensions to the C language
include("cuda/memory_shared.jl")
include("cuda/indexing.jl")
include("cuda/synchronization.jl")
include("cuda/warp_vote.jl")
include("cuda/warp_shuffle.jl")
include("cuda/output.jl")
include("cuda/assertion.jl")
include("cuda/memory_dynamic.jl")
include("cuda/atomics.jl")
include("cuda/misc.jl")
include("cuda/wmma.jl")

# functionality from libdevice
#
# > The libdevice library is a collection of NVVM bitcode functions that implement common
# > functions for NVIDIA GPU devices, including math primitives and bit-manipulation
# > functions. These functions are optimized for particular GPU architectures, and are
# > intended to be linked with an NVVM IR module during compilation to PTX.
include("cuda/math.jl")
# TODO: native mathematical functions, CUDA C programming guide" > "C language extensions"
#       https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
#       see /path/to/cuda/include/sm_20_intrinsics.h

# functionality from libcudadevrt
#
# The libcudadevrt library is a collection of PTX bitcode functions that implement parts of
# the CUDA API for execution on the device, such as device synchronization primitives,
# dynamic kernel APIs, etc.
using CEnum
include("cuda/libcudadevrt_common.jl")
include("cuda/libcudadevrt.jl")
include("cuda/cooperative_groups.jl")
include("cuda/dynamic_parallelism.jl")
