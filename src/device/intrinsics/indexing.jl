# Indexing and dimensions (B.4)

export
    threadIdx, blockDim, blockIdx, gridDim,
    warpsize

@generated function _index(::Val{name}, ::Val{range}) where {name, range}
    T_int32 = LLVM.Int32Type(jlctx[])

    # create function
    llvm_f, _ = create_function(T_int32)
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvm_f, "entry", jlctx[])
        position!(builder, entry)

        # call the indexing intrinsic
        intr_typ = LLVM.FunctionType(T_int32)
        intr = LLVM.Function(mod, "llvm.nvvm.read.ptx.sreg.$name", intr_typ)
        idx = call!(builder, intr)

        # attach range metadata
        range_metadata = MDNode([ConstantInt(Int32(range.start), jlctx[]),
                                 ConstantInt(Int32(range.stop), jlctx[])],
                                jlctx[])
        metadata(idx)[LLVM.MD_range] = range_metadata

        ret!(builder, idx)
    end

    call_function(llvm_f, UInt32)
end

for dim in (:x, :y, :z)
    # TODO: range per intrinsic & device
    range = 0:typemax(Int32)

    # Thread index
    fn = Symbol("threadIdx_$dim")
    intr = Symbol("tid.$dim")
    @eval @inline $fn() = _index($(Val(intr)), $(Val(range))) + UInt32(1)

    # Block size (#threads per block)
    fn = Symbol("blockDim_$dim")
    intr = Symbol("ntid.$dim")
    @eval @inline $fn() = _index($(Val(intr)), $(Val(range)))

    # Block index
    fn = Symbol("blockIdx_$dim")
    intr = Symbol("ctaid.$dim")
    @eval @inline $fn() = _index($(Val(intr)), $(Val(range))) + UInt32(1)

    # Grid size (#blocks per grid)
    fn = Symbol("gridDim_$dim")
    intr = Symbol("nctaid.$dim")
    @eval @inline $fn() = _index($(Val(intr)), $(Val(range)))
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
