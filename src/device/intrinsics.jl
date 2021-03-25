# wrappers for functionality provided by the CUDA toolkit

const overrides = quote end

macro device_override(ex)
    code = quote
        $GPUCompiler.@override($method_table, $ex)
    end
    if isdefined(Base.Experimental, Symbol("@overlay"))
        return esc(code)
    else
        push!(overrides.args, code)
        return
    end
end

macro device_function(ex)
    ex = macroexpand(__module__, ex)
    def = splitdef(ex)

    # generate a function that errors
    def[:body] = quote
        error("This function is not intended for use on the CPU")
    end

    esc(quote
        $(combinedef(def))
        @device_override $ex
    end)
end

# extensions to the C language
include("intrinsics/memory_shared.jl")
include("intrinsics/indexing.jl")
include("intrinsics/synchronization.jl")
include("intrinsics/warp_vote.jl")
include("intrinsics/warp_shuffle.jl")
include("intrinsics/output.jl")
include("intrinsics/assertion.jl")
include("intrinsics/memory_dynamic.jl")
include("intrinsics/atomics.jl")
include("intrinsics/misc.jl")
include("intrinsics/wmma.jl")

# functionality from libdevice
#
# > The libdevice library is a collection of NVVM bitcode functions that implement common
# > functions for NVIDIA GPU devices, including math primitives and bit-manipulation
# > functions. These functions are optimized for particular GPU architectures, and are
# > intended to be linked with an NVVM IR module during compilation to PTX.
include("intrinsics/math.jl")
# TODO: native mathematical functions, CUDA C programming guide" > "C language extensions"
#       https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
#       see /path/to/cuda/include/sm_20_intrinsics.h

# functionality from libcudadevrt
#
# The libcudadevrt library is a collection of PTX bitcode functions that implement parts of
# the CUDA API for execution on the device, such as device synchronization primitives,
# dynamic kernel APIs, etc.
using CEnum
include("intrinsics/libcudadevrt_common.jl")
include("intrinsics/libcudadevrt.jl")
include("intrinsics/cooperative_groups.jl")
include("intrinsics/dynamic_parallelism.jl")
