# Need https://github.com/JuliaLang/julia/pull/33970
# and  https://github.com/JuliaLang/julia/pull/34043
if VERSION < v"1.5-"
    exit()
end

using CUDA
if capability(device()) < v"7.0"
    exit()
end

### START
using Test

using CUDA

# Generate input matrices
a     = rand(Float16, (16, 16))
a_dev = CuArray(a)
b     = rand(Float16, (16, 16))
b_dev = CuArray(b)
c     = rand(Float32, (16, 16))
c_dev = CuArray(c)

# Allocate space for result
d_dev = similar(c_dev)

# Matrix multiply-accumulate kernel (D = A * B + C)
function kernel(a_dev, b_dev, c_dev, d_dev)
    a_frag = WMMA.llvm_wmma_load_a_col_m16n16k16_global_stride_f16(pointer(a_dev), 16)
    b_frag = WMMA.llvm_wmma_load_b_col_m16n16k16_global_stride_f16(pointer(b_dev), 16)
    c_frag = WMMA.llvm_wmma_load_c_col_m16n16k16_global_stride_f32(pointer(c_dev), 16)

    d_frag = WMMA.llvm_wmma_mma_col_col_m16n16k16_f32_f32(a_frag, b_frag, c_frag)

    WMMA.llvm_wmma_store_d_col_m16n16k16_global_stride_f32(pointer(d_dev), d_frag, 16)
    return
end

@cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)
@test all(isapprox.(a * b + c, Array(d_dev); rtol=0.01))
### END
