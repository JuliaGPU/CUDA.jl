# Indexing and dimensions (B.4)

export
    threadIdx, blockDim, blockIdx, gridDim, laneid, warpsize, active_mask, FULL_MASK

@generated function _index(::Val{name}, ::Val{range}) where {name, range}
    JuliaContext() do ctx
        T_int32 = LLVM.Int32Type(ctx)

        # create function
        llvm_f, _ = create_function(T_int32)
        mod = LLVM.parent(llvm_f)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            # call the indexing intrinsic
            intr_typ = LLVM.FunctionType(T_int32)
            intr = LLVM.Function(mod, "llvm.nvvm.read.ptx.sreg.$name", intr_typ)
            idx = call!(builder, intr)

            # attach range metadata
            range_metadata = MDNode([ConstantInt(Int32(range.start), ctx),
                                    ConstantInt(Int32(range.stop), ctx)],
                                    ctx)
            metadata(idx)[LLVM.MD_range] = range_metadata

            ret!(builder, idx)
        end

        call_function(llvm_f, UInt32)
    end
end

# XXX: these depend on the compute capability
#      https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
const max_block_size = (x=1024, y=1024, z=64)
const max_grid_size  = (x=2^31-1, y=65535, z=65535)

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

"""
    laneid()::UInt32

Returns the thread's lane within the warp.
"""
@inline laneid() = Int(ccall("llvm.nvvm.read.ptx.sreg.laneid", llvmcall, UInt32, ()))+UInt32(1)

"""
    active_mask()

Returns a 32-bit mask indicating which threads in a warp are active with the current
executing thread.
"""
@inline active_mask() = @asmcall("activemask.b32 \$0;", "=r", false, UInt32, Tuple{})

const FULL_MASK = 0xffffffff
