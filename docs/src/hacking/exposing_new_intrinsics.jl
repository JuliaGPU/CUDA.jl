# # Introduction

# * Adding new GPU intrinsics *

# In this tutorial we will expose some GPU intrinsics to allow directed rounding in fused-multiply-add (fma)
# floating point operation
# We start by identifying the intrinsic we want to expose; to do so, we read the PTX (Parallel Thread Execution) 
# documentation at [PTX - Floating Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions).
# In table 32, it is presented a summary of floating point operations: we can construct the intrinsic string from that.
# The FMA instruction for Float32 is presented as `{mad,fma}.rnd.f32`, where `rnd` can assume the values `.rnd = { .rn, .rz, .rm, .rp }`,
# where `rn` is round to nearest, `rz` round to zero, `rm` round to minus infinity, `rp` round to plus infinity.
# When building the intrinsic for the call, we need to change the type `.f64` with `.d` and `.f32` with `.f`
# Therefore, to call the rounded towards infinity `fma` for `.f64` we need to call the intrinsic `llvm.nvvm.fma.rp.d`
# Please remark that this is only possible if LLVM support the intrinsic; a source for those exposed by LLVM 
# may be found by searching the [LLVM repository](https://github.com/llvm/llvm-project). In in other cases you'd need @asmcall and inline PTX assembly.

fma_rp(x::Float64, y::Float64, z::Float64) = ccall("llvm.nvvm.fma.rp.d", llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)
fma(x::T, y::T, z::T, ::RoundingMode{:Up}) where {T <: Union{Float32, Float64}} = fma_rp(x, y, z)

# We inspect the PTX code
CUDA.code_ptx(fma_rp, Tuple{Float64,Float64,Float64})

# It is possible to see that the PTX code contains a call to the intrinsic `fma.rp.f64`; we add this function now 
# to src/device/intrins/math.jl

using CUDA
function test_fma!(out, x, y)
    I = threadIdx().x
    z = (2.0) ^ (-(I+53))

    out[I] = fma(x, y, z, RoundNearest)
    out[I+4] = fma(x, y, z, RoundToZero)
    out[I+8] = fma(x, y, z, RoundUp)
    out[I+12] = fma(x, y, z, RoundDown)

    return
end

# The first four entries of the output are Rounded to Nearest, the entries 5 to 8 are rounded towards zero,
# etc...

out_d = CuArray(zeros(16))
@cuda threads = 4 test_fma!(out_d, 1.0, 1.0)
out_h = Array(out_d)

out_d = CuArray(zeros(4))
@cuda threads = 4 test_fma!(out_d, -1.0, 1.0)
out_h = Array(out_d)

