# https://github.com/JuliaGPU/CUDA.jl/pull/1426

function kernel_wmma_f64_lowlevel(a_dev, b_dev, c_dev, d_dev)
    conf = WMMA.Config{8, 8, 4, Float64, RoundUp}
    
    # a_frag = WMMA.llvm_wmma_load_a_col_m8n8k4_global_stride_f64(pointer(a_dev), 8)
    # b_frag = WMMA.llvm_wmma_load_b_col_m8n8k4_global_stride_f64(pointer(b_dev), 4)
    # c_frag = WMMA.llvm_wmma_load_c_col_m8n8k4_global_stride_f64(pointer(c_dev), 8)

    a_frag = WMMA.load_a(pointer(a_dev), 8, ColMajor, conf)
    b_frag = WMMA.load_b(pointer(b_dev), 4, ColMajor, conf) 
    c_frag = WMMA.load_b(pointer(c_dev), 8, ColMajor, conf)

    d_frag = WMMA.llvm_wmma_mma(a_frag, b_frag, c_frag, conf)
    #d_frag = WMMA.llvm_wmma_mma_col_col_m8n8k4_f64(a_frag, b_frag, c_frag)
    #d_frag = WMMA.llvm_wmma_mma_col_col_m8n8k4_f64(a_frag, b_frag, c_frag, RoundToZero)
    #d_frag = WMMA.llvm_wmma_mma_col_col_m8n8k4_f64(a_frag, b_frag, c_frag, RoundUp)
    #d_frag = WMMA.llvm_wmma_mma_col_col_m8n8k4_f64(a_frag, b_frag, c_frag, RoundDown)
    #@cuprintln d_frag
    WWMA.store_d(pointer(d_dev), d_frag, 8, ColMajor, conf)
    #ccall("llvm.nvvm.wmma.m8n8k4.store.d.col.stride.f64", llvmcall, 
    #    Nothing, (Core.LLVMPtr{Float64, 1}, Float64, Float64,  Int32), 
    #    pointer(d_dev), d_frag[1], d_frag[2], 8)
    #WMMA.llvm_wmma_store_d_col_m8n8k4_global_stride_f64(pointer(d_dev), d_frag, 8)
    return nothing
end

function call_kernel()
    m = n = 8
    k = 4
    dtype_a = dtype_b = Float64
    dtype_c = dtype_d = Float64

    d_a = CUDA.rand(dtype_a, m, k)
    d_b = CUDA.rand(dtype_b, k, n)
    d_c = CUDA.rand(dtype_c, m, n)
    d_d = CUDA.zeros(dtype_d, m, n)

    CUDA.@sync @cuda kernel_wmma_f64_lowlevel(d_a, d_b, d_c, d_d)
    return nothing
end

#https://github.com/llvm/llvm-project/blob/main/clang/test/CodeGen/builtins-nvptx-mma.cu

