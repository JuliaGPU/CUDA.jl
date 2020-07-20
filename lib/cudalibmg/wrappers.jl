
mutable struct CudaLibMGDescriptor
    desc::cudaLibMgMatrixDesc_t

    function CudaLibMGDescriptor(a, grid; rowblocks = size(a, 1), colblocks = size(a, 2), elta = eltype(a) )
        desc = Ref{cudaLibMgMatrixDesc_t}()
        try
            cudaLibMgCreateMatrixDesc(desc, size(a, 1), size(a, 2), rowblocks, colblocks, cudaDataType(elta), grid)
        catch e
            println("size(A) = $(size(a)), rowblocks = $rowblocks, colblocks = $colblocks")
            flush(stdout)
            throw(e)
        end
        return new(desc[])
    end
end

Base.cconvert(::Type{cudaLibMgMatrixDesc_t}, obj::CudaLibMGDescriptor) = obj.desc

mutable struct CudaLibMGGrid
    desc::Ref{cudaLibMgGrid_t}

    function CudaLibMGGrid(num_row_devs, num_col_devs, deviceIds, mapping)
        desc = Ref{cudaLibMgGrid_t}()
        cudaLibMgCreateDeviceGrid(desc, num_row_devs, num_col_devs, deviceIds, mapping)
        return new(desc)
    end
end

Base.cconvert(::Type{cudaLibMgGrid_t}, obj::CudaLibMGGrid) = obj.desc[]
