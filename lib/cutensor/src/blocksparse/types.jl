## tensor

export CuTensorBS

## TODO add checks to see if size of data matches expected block size
mutable struct CuTensorBS{T, N}
    nonzero_data::Vector{<:CuArray}
    inds::Vector{Int}
    blocks_per_mode::Vector{Int}
    block_extents
    nonzero_block_coords

    function CuTensorBS{T, N}(nonzero_data::Vector{<:CuArray}, 
        blocks_per_mode::Vector{Int}, block_extents, nonzero_block_coords, inds::Vector) where {T<:Number, N}
        CuArrayT = eltype(nonzero_data)
        @assert eltype(CuArrayT) == T
        # @assert ndims(CuArrayT) == N
        @assert length(block_extents) == N
        new(nonzero_data, inds, blocks_per_mode, block_extents, nonzero_block_coords)
    end
end

function CuTensorBS(nonzero_data::Vector{<:CuArray{T}}, 
    blocks_per_mode, block_extents, nonzero_block_coords, inds::Vector) where {T<:Number}
    CuTensorBS{T,length(block_extents)}(nonzero_data, 
    blocks_per_mode, block_extents, nonzero_block_coords, inds)
end
# array interface
function Base.size(T::CuTensorBS)
    return tuple(sum.(T.block_extents)...)
end
Base.length(T::CuTensorBS) = prod(size(T))
nonzero_length(T::CuTensorBS) = sum(length.(T.nonzero_data))
Base.ndims(T::CuTensorBS) = length(T.inds)

Base.strides(T::CuTensorBS) = vcat([[st...] for st in strides.(T.nonzero_data)]...)
Base.eltype(T::CuTensorBS) = eltype(eltype(T.nonzero_data))

function block_extents(T::CuTensorBS)
    extents = Vector{Int64}() 
    
    for ex in T.block_extents
        extents = vcat(extents, ex...)
    end
    return extents
end

nblocks_per_mode(T::CuTensorBS) = T.blocks_per_mode

num_nonzero_blocks(T::CuTensorBS) = length(T.nonzero_block_coords)

## This function turns the tuple of the block coordinates into a single
## list of blocks
function list_nonzero_block_coords(T::CuTensorBS)
    block_list = Vector{Int64}()
    for block in T.nonzero_block_coords
        block_list = vcat(block_list, block...)
    end
    return block_list
end

# ## descriptor
mutable struct CuTensorBSDescriptor
    handle::cutensorBlockSparseTensorDescriptor_t
    # inner constructor handles creation and finalizer of the descriptor
    function CuTensorBSDescriptor(
        numModes,
        numNonZeroBlocks,
        numSectionsPerMode,
        extent,
        nonZeroCoordinates,
        stride,
        eltype)

        desc = Ref{cuTENSOR.cutensorBlockSparseTensorDescriptor_t}()
        cutensorCreateBlockSparseTensorDescriptor(handle(), desc, 
        numModes, numNonZeroBlocks, numSectionsPerMode, extent, nonZeroCoordinates,
        stride, eltype)

        obj = new(desc[])
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

function CuTensorBSDescriptor(
    numModes,
    numNonZeroBlocks,
    numSectionsPerMode,
    extent,
    nonZeroCoordinates,
    eltype)

    return CuTensorBSDescriptor(numModes, numNonZeroBlocks, numSectionsPerMode, extent, nonZeroCoordinates, C_NULL, eltype)
end

Base.show(io::IO, desc::CuTensorBSDescriptor) = @printf(io, "CuTensorBSDescriptor(%p)", desc.handle)

Base.unsafe_convert(::Type{cutensorBlockSparseTensorDescriptor_t}, obj::CuTensorBSDescriptor) = obj.handle

function unsafe_destroy!(obj::CuTensorBSDescriptor)
    cutensorDestroyBlockSparseTensorDescriptor(obj)
end

## Descriptor function for CuTensorBS type. Please overwrite for custom objects
function CuTensorBSDescriptor(A::CuTensorBS)
    numModes = Int32(ndims(A))
    numNonZeroBlocks = Int64(length(A.nonzero_block_coords))
    numSectionsPerMode = collect(Int32, A.blocks_per_mode)
    extent = block_extents(A)
    nonZeroCoordinates =  Int32.(vcat([[x...] for x in A.nonzero_block_coords]...) .- 1)
    st = strides(A)
    dataType = eltype(A)#convert(cuTENSOR.cutensorDataType_t, eltype(A))

    ## Right now assume stride is NULL. I am not sure if stride works, need to discuss with cuTENSOR team.
    CuTensorBSDescriptor(numModes, numNonZeroBlocks, 
    numSectionsPerMode, extent, nonZeroCoordinates, dataType)
end
