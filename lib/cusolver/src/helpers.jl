# cuSOLVER helper functions

## SparseQRInfo

mutable struct SparseQRInfo
    info::csrqrInfo_t

    function SparseQRInfo()
        info_ref = Ref{csrqrInfo_t}()
        cusolverSpCreateCsrqrInfo(info_ref)
        obj = new(info_ref[])
        finalizer(cusolverSpDestroyCsrqrInfo, obj)
        obj
    end
end

Base.unsafe_convert(::Type{csrqrInfo_t}, info::SparseQRInfo) = info.info


## SparseCholeskyInfo

mutable struct SparseCholeskyInfo
    info::csrcholInfo_t

    function SparseCholeskyInfo()
        info_ref = Ref{csrcholInfo_t}()
        cusolverSpCreateCsrcholInfo(info_ref)
        obj = new(info_ref[])
        finalizer(cusolverSpDestroyCsrcholInfo, obj)
        obj
    end
end

Base.unsafe_convert(::Type{csrcholInfo_t}, info::SparseCholeskyInfo) = info.info


## CuSolverParameters

mutable struct CuSolverParameters
    parameters::cusolverDnParams_t

    function CuSolverParameters()
        parameters_ref = Ref{cusolverDnParams_t}()
        cusolverDnCreateParams(parameters_ref)
        obj = new(parameters_ref[])
        finalizer(cusolverDnDestroyParams, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusolverDnParams_t}, params::CuSolverParameters) = params.parameters


## CuSolverIRSParameters

mutable struct CuSolverIRSParameters
    parameters::cusolverDnIRSParams_t

    function CuSolverIRSParameters()
        parameters_ref = Ref{cusolverDnIRSParams_t}()
        cusolverDnIRSParamsCreate(parameters_ref)
        obj = new(parameters_ref[])
        finalizer(cusolverDnIRSParamsDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusolverDnIRSParams_t}, params::CuSolverIRSParameters) = params.parameters

function get_info(params::CuSolverIRSParameters, field::Symbol)
    if field == :maxiters
        maxiters = Ref{Cint}()
        cusolverDnIRSParamsGetMaxIters(params, maxiters)
        return maxiters[]
    else
        error("The information $field is incorrect.")
    end
end


## CuSolverIRSInformation

mutable struct CuSolverIRSInformation
    information::cusolverDnIRSInfos_t

    function CuSolverIRSInformation()
        info_ref = Ref{cusolverDnIRSInfos_t}()
        cusolverDnIRSInfosCreate(info_ref)
        obj = new(info_ref[])
        finalizer(cusolverDnIRSInfosDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusolverDnIRSInfos_t}, info::CuSolverIRSInformation) = info.information

function get_info(info::CuSolverIRSInformation, field::Symbol)
    if field == :niters
        niters = Ref{Cint}()
        cusolverDnIRSInfosGetNiters(info, niters)
        return niters[]
    elseif field == :outer_niters
        outer_niters = Ref{Cint}()
        cusolverDnIRSInfosGetOuterNiters(info, outer_niters)
        return outer_niters[]
    # elseif field == :residual_history
    #     residual_history = Ref{Ptr{Cvoid}
    #     cusolverDnIRSInfosGetResidualHistory(info, residual_history)
    #     return residual_history[]
    elseif field == :maxiters
        maxiters = Ref{Cint}()
        cusolverDnIRSInfosGetMaxIters(info, maxiters)
        return maxiters[]
    else
        error("The information $field is incorrect.")
    end
end


## MatrixDescriptor

mutable struct MatrixDescriptor
    desc::cudaLibMgMatrixDesc_t

    function MatrixDescriptor(a, grid; rowblocks = size(a, 1), colblocks = size(a, 2), elta=eltype(a) )
        desc = Ref{cudaLibMgMatrixDesc_t}()
        cusolverMgCreateMatrixDesc(desc, size(a, 1), size(a, 2), rowblocks, colblocks, elta, grid)
        return new(desc[])
    end
end

Base.unsafe_convert(::Type{cudaLibMgMatrixDesc_t}, obj::MatrixDescriptor) = obj.desc


## DeviceGrid

mutable struct DeviceGrid
    desc::cudaLibMgGrid_t

    function DeviceGrid(num_row_devs, num_col_devs, deviceIds, mapping)
        @assert num_row_devs == 1 "Only 1-D column block cyclic is supported, so numRowDevices must be equal to 1."
        desc = Ref{cudaLibMgGrid_t}()
        cusolverMgCreateDeviceGrid(desc, num_row_devs, num_col_devs, deviceIds, mapping)
        return new(desc[])
    end
end

Base.unsafe_convert(::Type{cudaLibMgGrid_t}, obj::DeviceGrid) = obj.desc
