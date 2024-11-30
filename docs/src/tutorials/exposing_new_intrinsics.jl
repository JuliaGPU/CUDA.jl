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

fma_rp(x::Float64, y::Float64, z::Float64) = ccall("llvm.nvvm.fma.rp.d", llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)

# We inspect the PTX code
CUDA.code_ptx(fma_rp, Tuple{Float64, Float64, Float64})

# It is possible to see that the PTX code contains a call to the intrinsic `fma.rp.f64`; we add this function now 
# to src/device/intrins/math.jl

function test_fma!(out, x, y, z)
    I = threadIdx().x
    if I%4 == 0
        out[I] = CUDA.fma_rn(x, y, z)
    elseif I%4 ==1 
        out[I] = CUDA.fma_rz(x, y, z)
    elseif I%4 ==2 
        out[I] = CUDA.fma_rm(x, y, z)
    elseif I%4 ==3 
        out[I] = CUDA.fma_rp(x, y, z)
    end
    return 
end

# The first thread computes round to nearest and stores in the first entry, the second thread computes
# round towards zero and store in the second, the third rounds towards minus infinity, the fourth towards plus infinity

out_d = CuArray(zeros(4))
@cuda threads = 4 test_fma!(out_d, 1.0, 1.0, 2^(-54))
out_h = Array(out_d)

out_d = CuArray(zeros(4))
@cuda threads = 4 test_fma!(out_d, -1.0, 1.0, 2^(-54))
out_h = Array(out_d)

# The binary operations as add, sub, mul, div have been implemented through a macro

function test_add!(out, x, y)
    I = threadIdx().x
    if I%4 == 0
        out[I] = CUDA.add_rn(x, y)
    elseif I%4 ==1 
        out[I] = CUDA.add_rz(x, y)
    elseif I%4 ==2 
        out[I] = CUDA.add_rm(x, y)
    elseif I%4 ==3 
        out[I] = CUDA.add_rp(x, y)
    end
    return 
end

out_d = CuArray(zeros(4))
@cuda threads = 4 test_add!(out_d, 1.0, 2^(-53))
out_h = Array(out_d)

function test_sub!(out, x, y)
    I = threadIdx().x
    if I%4 == 0
        out[I] = CUDA.sub_rn(x, y)
    elseif I%4 ==1 
        out[I] = CUDA.sub_rz(x, y)
    elseif I%4 ==2 
        out[I] = CUDA.sub_rm(x, y)
    elseif I%4 ==3 
        out[I] = CUDA.sub_rp(x, y)
    end
    return 
end

out_d = CuArray(zeros(4))
@cuda threads = 4 test_sub!(out_d, 1.0, 2^(-53))
out_h = Array(out_d)
