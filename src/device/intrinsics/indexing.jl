# Indexing and dimensions (B.4)

export
    threadIdx, blockDim, blockIdx, gridDim,
    warpsize

@generated function _index(::Val{name}, ::Val{range}) where {name, range}
    T_int32 = LLVM.Int32Type(JuliaContext())

    # create function
    llvm_f, _ = create_function(T_int32)
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        # call the indexing intrinsic
        intr_typ = LLVM.FunctionType(T_int32)
        intr = LLVM.Function(mod, "llvm.nvvm.read.ptx.sreg.$name", intr_typ)
        idx = call!(builder, intr)

        # attach range metadata
        range_metadata = MDNode([ConstantInt(Int32(range.start), JuliaContext()),
                                 ConstantInt(Int32(range.stop), JuliaContext())],
                                JuliaContext())
        metadata(idx)[LLVM.MD_range] = range_metadata

        ret!(builder, idx)
    end

    call_function(llvm_f, UInt32)
end

# TODO: look these up for the current device (using contextual dispatch).
#       for now, these values are based on the Volta V100 GPU.
const max_block_size = (x=1024, y=1024, z=1024)
const max_grid_size  = (x=2147483647, y=65535, z=65535)

for dim in (:x, :y, :z)
    # Thread index
    fn = Symbol("threadIdx_$dim")
    intr = Symbol("tid.$dim")
    @eval @inline $fn() = Int(_index($(Val(intr)), $(Val(0:max_block_size[dim]-1)))) + 1

    # Block size (#threads per block)
    fn = Symbol("blockDim_$dim")
    intr = Symbol("ntid.$dim")
    @eval @inline $fn() = Int(_index($(Val(intr)), $(Val(1:max_block_size[dim]))))

    # Block index
    fn = Symbol("blockIdx_$dim")
    intr = Symbol("ctaid.$dim")
    @eval @inline $fn() = Int(_index($(Val(intr)), $(Val(0:max_grid_size[dim]-1)))) + 1

    # Grid size (#blocks per grid)
    fn = Symbol("gridDim_$dim")
    intr = Symbol("nctaid.$dim")
    @eval @inline $fn() = Int(_index($(Val(intr)), $(Val(1:max_grid_size[dim]))))
end

"""
    gridDim()::CuDim3

Returns the dimensions of the grid.
"""
@inline gridDim() =   (x=gridDim_x(),   y=gridDim_y(),   z=gridDim_z())

"""
    blockIdx()::CuDim3

Returns the block index within the grid.
"""
@inline blockIdx() =  (x=blockIdx_x(),  y=blockIdx_y(),  z=blockIdx_z())

"""
    blockDim()::CuDim3

Returns the dimensions of the block.
"""
@inline blockDim() =  (x=blockDim_x(),  y=blockDim_y(),  z=blockDim_z())

"""
    threadIdx()::CuDim3

Returns the thread index within the block. 
"""
@inline threadIdx() = (x=threadIdx_x(), y=threadIdx_y(), z=threadIdx_z())

"""
    warpsize()::UInt32

Returns the warp size (in threads).
"""
@inline warpsize() = Int(ccall("llvm.nvvm.read.ptx.sreg.warpsize", llvmcall, UInt32, ()))
