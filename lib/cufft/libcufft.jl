using CEnum

# CUFFT uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUFFT_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUFFTError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUFFT_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUFFT_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end


@cenum cufftResult_t::UInt32 begin
    CUFFT_SUCCESS = 0
    CUFFT_INVALID_PLAN = 1
    CUFFT_ALLOC_FAILED = 2
    CUFFT_INVALID_TYPE = 3
    CUFFT_INVALID_VALUE = 4
    CUFFT_INTERNAL_ERROR = 5
    CUFFT_EXEC_FAILED = 6
    CUFFT_SETUP_FAILED = 7
    CUFFT_INVALID_SIZE = 8
    CUFFT_UNALIGNED_DATA = 9
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10
    CUFFT_INVALID_DEVICE = 11
    CUFFT_PARSE_ERROR = 12
    CUFFT_NO_WORKSPACE = 13
    CUFFT_NOT_IMPLEMENTED = 14
    CUFFT_LICENSE_ERROR = 15
    CUFFT_NOT_SUPPORTED = 16
end

const cufftResult = cufftResult_t

const cufftReal = Cfloat

const cufftDoubleReal = Cdouble

const cufftComplex = cuComplex

const cufftDoubleComplex = cuDoubleComplex

@cenum cufftType_t::UInt32 begin
    CUFFT_R2C = 42
    CUFFT_C2R = 44
    CUFFT_C2C = 41
    CUFFT_D2Z = 106
    CUFFT_Z2D = 108
    CUFFT_Z2Z = 105
end

const cufftType = cufftType_t

@cenum cufftCompatibility_t::UInt32 begin
    CUFFT_COMPATIBILITY_FFTW_PADDING = 1
end

const cufftCompatibility = cufftCompatibility_t

const cufftHandle = Cint

@checked function cufftPlan1d(plan, nx, type, batch)
        initialize_context()
        ccall((:cufftPlan1d, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, cufftType, Cint), plan, nx, type, batch)
    end

@checked function cufftPlan2d(plan, nx, ny, type)
        initialize_context()
        ccall((:cufftPlan2d, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, Cint, cufftType), plan, nx, ny, type)
    end

@checked function cufftPlan3d(plan, nx, ny, nz, type)
        initialize_context()
        ccall((:cufftPlan3d, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, Cint, Cint, cufftType), plan, nx, ny, nz, type)
    end

@checked function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)
        initialize_context()
        ccall((:cufftPlanMany, libcufft), cufftResult, (Ptr{cufftHandle}, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint, cufftType, Cint), plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)
    end

@checked function cufftMakePlan1d(plan, nx, type, batch, workSize)
        initialize_context()
        ccall((:cufftMakePlan1d, libcufft), cufftResult, (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}), plan, nx, type, batch, workSize)
    end

@checked function cufftMakePlan2d(plan, nx, ny, type, workSize)
        initialize_context()
        ccall((:cufftMakePlan2d, libcufft), cufftResult, (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}), plan, nx, ny, type, workSize)
    end

@checked function cufftMakePlan3d(plan, nx, ny, nz, type, workSize)
        initialize_context()
        ccall((:cufftMakePlan3d, libcufft), cufftResult, (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}), plan, nx, ny, nz, type, workSize)
    end

@checked function cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
        initialize_context()
        ccall((:cufftMakePlanMany, libcufft), cufftResult, (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint, cufftType, Cint, Ptr{Csize_t}), plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
    end

@checked function cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
        initialize_context()
        ccall((:cufftMakePlanMany64, libcufft), cufftResult, (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong, Clonglong, Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong, Ptr{Csize_t}), plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
    end

@checked function cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
        initialize_context()
        ccall((:cufftGetSizeMany64, libcufft), cufftResult, (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong, Clonglong, Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong, Ptr{Csize_t}), plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
    end

@checked function cufftEstimate1d(nx, type, batch, workSize)
        initialize_context()
        ccall((:cufftEstimate1d, libcufft), cufftResult, (Cint, cufftType, Cint, Ptr{Csize_t}), nx, type, batch, workSize)
    end

@checked function cufftEstimate2d(nx, ny, type, workSize)
        initialize_context()
        ccall((:cufftEstimate2d, libcufft), cufftResult, (Cint, Cint, cufftType, Ptr{Csize_t}), nx, ny, type, workSize)
    end

@checked function cufftEstimate3d(nx, ny, nz, type, workSize)
        initialize_context()
        ccall((:cufftEstimate3d, libcufft), cufftResult, (Cint, Cint, Cint, cufftType, Ptr{Csize_t}), nx, ny, nz, type, workSize)
    end

@checked function cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
        initialize_context()
        ccall((:cufftEstimateMany, libcufft), cufftResult, (Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint, cufftType, Cint, Ptr{Csize_t}), rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
    end

@checked function cufftCreate(handle)
        initialize_context()
        ccall((:cufftCreate, libcufft), cufftResult, (Ptr{cufftHandle},), handle)
    end

@checked function cufftGetSize1d(handle, nx, type, batch, workSize)
        initialize_context()
        ccall((:cufftGetSize1d, libcufft), cufftResult, (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}), handle, nx, type, batch, workSize)
    end

@checked function cufftGetSize2d(handle, nx, ny, type, workSize)
        initialize_context()
        ccall((:cufftGetSize2d, libcufft), cufftResult, (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}), handle, nx, ny, type, workSize)
    end

@checked function cufftGetSize3d(handle, nx, ny, nz, type, workSize)
        initialize_context()
        ccall((:cufftGetSize3d, libcufft), cufftResult, (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}), handle, nx, ny, nz, type, workSize)
    end

@checked function cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea)
        initialize_context()
        ccall((:cufftGetSizeMany, libcufft), cufftResult, (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint, cufftType, Cint, Ptr{Csize_t}), handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea)
    end

@checked function cufftGetSize(handle, workSize)
        initialize_context()
        ccall((:cufftGetSize, libcufft), cufftResult, (cufftHandle, Ptr{Csize_t}), handle, workSize)
    end

@checked function cufftSetWorkArea(plan, workArea)
        initialize_context()
        ccall((:cufftSetWorkArea, libcufft), cufftResult, (cufftHandle, CuPtr{Cvoid}), plan, workArea)
    end

@checked function cufftSetAutoAllocation(plan, autoAllocate)
        initialize_context()
        ccall((:cufftSetAutoAllocation, libcufft), cufftResult, (cufftHandle, Cint), plan, autoAllocate)
    end

@checked function cufftExecC2C(plan, idata, odata, direction)
        initialize_context()
        ccall((:cufftExecC2C, libcufft), cufftResult, (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftComplex}, Cint), plan, idata, odata, direction)
    end

@checked function cufftExecR2C(plan, idata, odata)
        initialize_context()
        ccall((:cufftExecR2C, libcufft), cufftResult, (cufftHandle, CuPtr{cufftReal}, CuPtr{cufftComplex}), plan, idata, odata)
    end

@checked function cufftExecC2R(plan, idata, odata)
        initialize_context()
        ccall((:cufftExecC2R, libcufft), cufftResult, (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftReal}), plan, idata, odata)
    end

@checked function cufftExecZ2Z(plan, idata, odata, direction)
        initialize_context()
        ccall((:cufftExecZ2Z, libcufft), cufftResult, (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex}, Cint), plan, idata, odata, direction)
    end

@checked function cufftExecD2Z(plan, idata, odata)
        initialize_context()
        ccall((:cufftExecD2Z, libcufft), cufftResult, (cufftHandle, CuPtr{cufftDoubleReal}, CuPtr{cufftDoubleComplex}), plan, idata, odata)
    end

@checked function cufftExecZ2D(plan, idata, odata)
        initialize_context()
        ccall((:cufftExecZ2D, libcufft), cufftResult, (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleReal}), plan, idata, odata)
    end

@checked function cufftSetStream(plan, stream)
        initialize_context()
        ccall((:cufftSetStream, libcufft), cufftResult, (cufftHandle, cudaStream_t), plan, stream)
    end

@checked function cufftDestroy(plan)
        initialize_context()
        ccall((:cufftDestroy, libcufft), cufftResult, (cufftHandle,), plan)
    end

@checked function cufftGetVersion(version)
        ccall((:cufftGetVersion, libcufft), cufftResult, (Ptr{Cint},), version)
    end

@checked function cufftGetProperty(type, value)
        ccall((:cufftGetProperty, libcufft), cufftResult, (libraryPropertyType, Ptr{Cint}), type, value)
    end

# Skipping MacroDefinition: CUFFTAPI __attribute__ ( ( visibility ( "default" ) ) )

const MAX_CUFFT_ERROR = 0x11

const CUFFT_FORWARD = -1

const CUFFT_INVERSE = 1

const CUFFT_COMPATIBILITY_DEFAULT = CUFFT_COMPATIBILITY_FFTW_PADDING

const MAX_SHIM_RANK = 3

