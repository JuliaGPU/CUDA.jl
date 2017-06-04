# Indexing and dimensions (B.4)

export
    threadIdx, blockDim, blockIdx, gridDim,
    warpsize

for dim in (:x, :y, :z)
    # Thread index
    fn = Symbol("threadIdx_$dim")
    @eval @inline $fn() = (ccall($"llvm.nvvm.read.ptx.sreg.tid.$dim", llvmcall, UInt32, ()))+UInt32(1)

    # Block size (#threads per block)
    fn = Symbol("blockDim_$dim")
    @eval @inline $fn() =  ccall($"llvm.nvvm.read.ptx.sreg.ntid.$dim", llvmcall, UInt32, ())

    # Block index
    fn = Symbol("blockIdx_$dim")
    @eval @inline $fn() = (ccall($"llvm.nvvm.read.ptx.sreg.ctaid.$dim", llvmcall, UInt32, ()))+UInt32(1)

    # Grid size (#blocks per grid)
    fn = Symbol("gridDim_$dim")
    @eval @inline $fn() =  ccall($"llvm.nvvm.read.ptx.sreg.nctaid.$dim", llvmcall, UInt32, ())
end

"""
    gridDim()::CuDim3

Returns the dimensions of the grid.
"""
@inline gridDim() =   CUDAdrv.CuDim3(gridDim_x(),   gridDim_y(),   gridDim_z())

"""
    blockIdx()::CuDim3

Returns the block index within the grid.
"""
@inline blockIdx() =  CUDAdrv.CuDim3(blockIdx_x(),  blockIdx_y(),  blockIdx_z())

"""
    blockDim()::CuDim3

Returns the dimensions of the block.
"""
@inline blockDim() =  CUDAdrv.CuDim3(blockDim_x(),  blockDim_y(),  blockDim_z())

"""
    threadIdx()::CuDim3

Returns the thread index within the block. 
"""
@inline threadIdx() = CUDAdrv.CuDim3(threadIdx_x(), threadIdx_y(), threadIdx_z())

"""
    warpsize()::UInt32

Returns the warp size (in threads).
"""
@inline warpsize() = ccall("llvm.nvvm.read.ptx.sreg.warpsize", llvmcall, UInt32, ())
