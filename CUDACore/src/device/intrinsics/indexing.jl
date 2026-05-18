# Indexing and dimensions (B.4)

export
    threadIdx, blockDim, blockIdx, gridDim, blockIdxInCluster, clusterDim, clusterIdx, gridClusterDim,
    linearBlockIdxInCluster, linearClusterSize,
    laneid, lanemask, warpsize, active_mask, FULL_MASK

@device_function @generated function _index(::Val{name}, ::Val{range}) where {name, range}
    @dispose ctx=Context() begin
        T_int32 = LLVM.Int32Type()

        # create function
        llvm_f, _ = create_function(T_int32)
        mod = LLVM.parent(llvm_f)

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            # call the indexing intrinsic
            intr_typ = LLVM.FunctionType(T_int32)
            intr = LLVM.Function(mod, "llvm.nvvm.read.ptx.sreg.$name", intr_typ)
            idx = call!(builder, intr_typ, intr)

            # attach range metadata
            range_metadata = MDNode([ConstantInt(range.start % Int32),
                                     ConstantInt((range.stop + 1) % Int32)])
            metadata(idx)[LLVM.MD_range] = range_metadata

            ret!(builder, idx)
        end

        call_function(llvm_f, Int32)
    end
end

# XXX: these depend on the compute capability
#      https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
const max_block_size = (x=1024, y=1024, z=64)
const max_block_length = 1024
const max_grid_size  = (x=2^31-1, y=65535, z=65535)
# maximum guaranteed linear dimension is 8, but 16 is possible on Hopper
# https://forums.developer.nvidia.com/t/cluster-size-limitation/279795
const max_cluster_size = (x=16, y=16, z=16)
const max_cluster_length = 16

for dim in (:x, :y, :z)
    # Thread index in block
    fn = Symbol("threadIdx_$dim")
    intr = Symbol("tid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(0:max_block_size[dim]-1))) + 1i32

    # Block size (#threads per block)
    fn = Symbol("blockDim_$dim")
    intr = Symbol("ntid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(1:max_block_size[dim])))

    # Block index in grid
    fn = Symbol("blockIdx_$dim")
    intr = Symbol("ctaid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(0:max_grid_size[dim]-1))) + 1i32

    # Grid size (#blocks per grid)
    fn = Symbol("gridDim_$dim")
    intr = Symbol("nctaid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(1:max_grid_size[dim])))

    # Block index in cluster
    fn = Symbol("blockIdxInCluster_$dim")
    intr = Symbol("cluster.ctaid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(0:max_cluster_size[dim]-1))) + 1i32

    # Cluster size (#blocks per cluster)
    fn = Symbol("clusterDim_$dim")
    intr = Symbol("cluster.nctaid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(1:max_cluster_size[dim])))

    # Cluster index in grid
    fn = Symbol("clusterIdx_$dim")
    intr = Symbol("clusterid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(0:max_grid_size[dim]-1))) + 1i32

    # Grid size in clusters (#clusters per grid)
    fn = Symbol("gridClusterDim_$dim")
    intr = Symbol("nclusterid.$dim")
    @eval @device_function @inline $fn() = _index($(Val(intr)), $(Val(1:max_grid_size[dim])))
end

@device_functions begin

@doc """
    threadIdx()::NamedTuple

Returns the thread index within the block as a `NamedTuple` with keys `x`, `y`, and `z`.
These indices are 1-based, unlike the `threadIdx` built-in variable in the C/C++ extension which is 0-based.
""" threadIdx
@inline threadIdx() = (x=threadIdx_x(), y=threadIdx_y(), z=threadIdx_z())

@doc """
    blockDim()::NamedTuple

Returns the dimensions (in threads) of the block as a `NamedTuple` with keys `x`, `y`, and `z`.
Unlike the `*Idx` intrinsics, `blockDim` returns the same value as its C/C++ extension counterpart.
""" blockDim
@inline blockDim() = (x=blockDim_x(), y=blockDim_y(), z=blockDim_z())

@doc """
    blockIdx()::NamedTuple

Returns the block index within the grid as a `NamedTuple` with keys `x`, `y`, and `z`.
These indices are 1-based, unlike the `blockIdx` built-in variable in the C/C++ extension which is 0-based.
""" blockIdx
@inline blockIdx() = (x=blockIdx_x(), y=blockIdx_y(), z=blockIdx_z())

@doc """
    gridDim()::NamedTuple

Returns the dimensions (in blocks) of the grid as a `NamedTuple` with keys `x`, `y`, and `z`.
Unlike the `*Idx` intrinsics, `gridDim` returns the same value as its C/C++ extension counterpart.
""" gridDim
@inline gridDim() = (x=gridDim_x(), y=gridDim_y(), z=gridDim_z())

@doc """
    blockIdxInCluster()::NamedTuple

Returns the block index within the cluster as a `NamedTuple` with keys `x`, `y`, and `z`.
These indices are 1-based.
""" blockIdxInCluster
@inline blockIdxInCluster() = (x=blockIdxInCluster_x(), y=blockIdxInCluster_y(), z=blockIdxInCluster_z())

@doc """
    clusterDim()::NamedTuple

Returns the dimensions (in blocks) of the cluster as a `NamedTuple` with keys `x`, `y`, and `z`.
""" clusterDim
@inline clusterDim() = (x=clusterDim_x(), y=clusterDim_y(), z=clusterDim_z())

@doc """
    clusterIdx()::NamedTuple

Returns the cluster index within the grid as a `NamedTuple` with keys `x`, `y`, and `z`.
These indices are 1-based.
""" clusterIdx
@inline clusterIdx() = (x=clusterIdx_x(), y=clusterIdx_y(), z=clusterIdx_z())

@doc """
    gridClusterDim()::NamedTuple

Returns the dimensions (in clusters) of the grid as a `NamedTuple` with keys `x`, `y`, and `z`.
""" gridClusterDim
@inline gridClusterDim() = (x=gridClusterDim_x(), y=gridClusterDim_y(), z=gridClusterDim_z())

@doc """
    linearBlockIdxInCluster()::Int32

Returns the linear block index within the cluster.
These indices are 1-based.
""" linearBlockIdxInCluster
@eval @inline $(:linearBlockIdxInCluster)() = _index($(Val(Symbol("cluster.ctarank"))), $(Val(0:max_cluster_length-1))) + 1i32

@doc """
    linearClusterSize()::Int32

Returns the linear cluster size (in blocks).
""" linearClusterSize
@eval @inline $(:linearClusterSize)() = _index($(Val(Symbol("cluster.nctarank"))), $(Val(1:max_cluster_length)))

@doc """
    warpsize()::Int32

Returns the warp size (in threads).
This corresponds to the `warpSize` built-in variable in the C/C++ extension.
""" warpsize
@inline warpsize() = ccall("llvm.nvvm.read.ptx.sreg.warpsize", llvmcall, Int32, ())

@doc """
    laneid()::Int32

Returns the thread's lane within the warp.
This ID is 1-based.
""" laneid
@inline laneid() = ccall("llvm.nvvm.read.ptx.sreg.laneid", llvmcall, Int32, ()) + 1i32

@doc """
    lanemask(pred)::UInt32

Returns a 32-bit mask indicating which threads in a warp satisfy the given predicate.
Supported predicates are `==`, `<`, `<=`, `>=`, and `>`.
""" lanemask
@inline function lanemask(pred::F) where F
    if pred === Base.:(==)
        ccall("llvm.nvvm.read.ptx.sreg.lanemask.eq", llvmcall, UInt32, ())
    elseif pred === Base.:(<)
        ccall("llvm.nvvm.read.ptx.sreg.lanemask.lt", llvmcall, UInt32, ())
    elseif pred === Base.:(<=)
        ccall("llvm.nvvm.read.ptx.sreg.lanemask.le", llvmcall, UInt32, ())
    elseif pred === Base.:(>=)
        ccall("llvm.nvvm.read.ptx.sreg.lanemask.ge", llvmcall, UInt32, ())
    elseif pred === Base.:(>)
        ccall("llvm.nvvm.read.ptx.sreg.lanemask.gt", llvmcall, UInt32, ())
    else
        throw(ArgumentError("invalid lanemask function"))
    end
end

@doc """
    active_mask()

Returns a 32-bit mask indicating which threads in a warp are active with the current
executing thread.
""" active_mask
@inline active_mask() = @asmcall("activemask.b32 \$0;", "=r", false, UInt32, Tuple{})

end

@doc """
    FULL_MASK

A 32-bit mask indicating that all threads in a warp are active.
""" FULL_MASK
const FULL_MASK = 0xffffffff
