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
        res = @retry_reclaim err -> $check $(esc(ex))
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
    @ccall libcufft.cufftPlan1d(plan::Ptr{cufftHandle}, nx::Cint, type::cufftType,
                                batch::Cint)::cufftResult
end

@checked function cufftPlan2d(plan, nx, ny, type)
    initialize_context()
    @ccall libcufft.cufftPlan2d(plan::Ptr{cufftHandle}, nx::Cint, ny::Cint,
                                type::cufftType)::cufftResult
end

@checked function cufftPlan3d(plan, nx, ny, nz, type)
    initialize_context()
    @ccall libcufft.cufftPlan3d(plan::Ptr{cufftHandle}, nx::Cint, ny::Cint, nz::Cint,
                                type::cufftType)::cufftResult
end

@checked function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride,
                                odist, type, batch)
    initialize_context()
    @ccall libcufft.cufftPlanMany(plan::Ptr{cufftHandle}, rank::Cint, n::Ptr{Cint},
                                  inembed::Ptr{Cint}, istride::Cint, idist::Cint,
                                  onembed::Ptr{Cint}, ostride::Cint, odist::Cint,
                                  type::cufftType, batch::Cint)::cufftResult
end

@checked function cufftMakePlan1d(plan, nx, type, batch, workSize)
    initialize_context()
    @ccall libcufft.cufftMakePlan1d(plan::cufftHandle, nx::Cint, type::cufftType,
                                    batch::Cint, workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftMakePlan2d(plan, nx, ny, type, workSize)
    initialize_context()
    @ccall libcufft.cufftMakePlan2d(plan::cufftHandle, nx::Cint, ny::Cint, type::cufftType,
                                    workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftMakePlan3d(plan, nx, ny, nz, type, workSize)
    initialize_context()
    @ccall libcufft.cufftMakePlan3d(plan::cufftHandle, nx::Cint, ny::Cint, nz::Cint,
                                    type::cufftType, workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed,
                                    ostride, odist, type, batch, workSize)
    initialize_context()
    @ccall libcufft.cufftMakePlanMany(plan::cufftHandle, rank::Cint, n::Ptr{Cint},
                                      inembed::Ptr{Cint}, istride::Cint, idist::Cint,
                                      onembed::Ptr{Cint}, ostride::Cint, odist::Cint,
                                      type::cufftType, batch::Cint,
                                      workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed,
                                      ostride, odist, type, batch, workSize)
    initialize_context()
    @ccall libcufft.cufftMakePlanMany64(plan::cufftHandle, rank::Cint, n::Ptr{Clonglong},
                                        inembed::Ptr{Clonglong}, istride::Clonglong,
                                        idist::Clonglong, onembed::Ptr{Clonglong},
                                        ostride::Clonglong, odist::Clonglong,
                                        type::cufftType, batch::Clonglong,
                                        workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed,
                                     ostride, odist, type, batch, workSize)
    initialize_context()
    @ccall libcufft.cufftGetSizeMany64(plan::cufftHandle, rank::Cint, n::Ptr{Clonglong},
                                       inembed::Ptr{Clonglong}, istride::Clonglong,
                                       idist::Clonglong, onembed::Ptr{Clonglong},
                                       ostride::Clonglong, odist::Clonglong,
                                       type::cufftType, batch::Clonglong,
                                       workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftEstimate1d(nx, type, batch, workSize)
    initialize_context()
    @ccall libcufft.cufftEstimate1d(nx::Cint, type::cufftType, batch::Cint,
                                    workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftEstimate2d(nx, ny, type, workSize)
    initialize_context()
    @ccall libcufft.cufftEstimate2d(nx::Cint, ny::Cint, type::cufftType,
                                    workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftEstimate3d(nx, ny, nz, type, workSize)
    initialize_context()
    @ccall libcufft.cufftEstimate3d(nx::Cint, ny::Cint, nz::Cint, type::cufftType,
                                    workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                                    odist, type, batch, workSize)
    initialize_context()
    @ccall libcufft.cufftEstimateMany(rank::Cint, n::Ptr{Cint}, inembed::Ptr{Cint},
                                      istride::Cint, idist::Cint, onembed::Ptr{Cint},
                                      ostride::Cint, odist::Cint, type::cufftType,
                                      batch::Cint, workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftCreate(handle)
    initialize_context()
    @ccall libcufft.cufftCreate(handle::Ptr{cufftHandle})::cufftResult
end

@checked function cufftGetSize1d(handle, nx, type, batch, workSize)
    initialize_context()
    @ccall libcufft.cufftGetSize1d(handle::cufftHandle, nx::Cint, type::cufftType,
                                   batch::Cint, workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftGetSize2d(handle, nx, ny, type, workSize)
    initialize_context()
    @ccall libcufft.cufftGetSize2d(handle::cufftHandle, nx::Cint, ny::Cint, type::cufftType,
                                   workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftGetSize3d(handle, nx, ny, nz, type, workSize)
    initialize_context()
    @ccall libcufft.cufftGetSize3d(handle::cufftHandle, nx::Cint, ny::Cint, nz::Cint,
                                   type::cufftType, workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed,
                                   ostride, odist, type, batch, workArea)
    initialize_context()
    @ccall libcufft.cufftGetSizeMany(handle::cufftHandle, rank::Cint, n::Ptr{Cint},
                                     inembed::Ptr{Cint}, istride::Cint, idist::Cint,
                                     onembed::Ptr{Cint}, ostride::Cint, odist::Cint,
                                     type::cufftType, batch::Cint,
                                     workArea::Ptr{Csize_t})::cufftResult
end

@checked function cufftGetSize(handle, workSize)
    initialize_context()
    @ccall libcufft.cufftGetSize(handle::cufftHandle, workSize::Ptr{Csize_t})::cufftResult
end

@checked function cufftSetWorkArea(plan, workArea)
    initialize_context()
    @ccall libcufft.cufftSetWorkArea(plan::cufftHandle, workArea::CuPtr{Cvoid})::cufftResult
end

@checked function cufftSetAutoAllocation(plan, autoAllocate)
    initialize_context()
    @ccall libcufft.cufftSetAutoAllocation(plan::cufftHandle,
                                           autoAllocate::Cint)::cufftResult
end

@checked function cufftExecC2C(plan, idata, odata, direction)
    initialize_context()
    @ccall libcufft.cufftExecC2C(plan::cufftHandle, idata::CuPtr{cufftComplex},
                                 odata::CuPtr{cufftComplex}, direction::Cint)::cufftResult
end

@checked function cufftExecR2C(plan, idata, odata)
    initialize_context()
    @ccall libcufft.cufftExecR2C(plan::cufftHandle, idata::CuPtr{cufftReal},
                                 odata::CuPtr{cufftComplex})::cufftResult
end

@checked function cufftExecC2R(plan, idata, odata)
    initialize_context()
    @ccall libcufft.cufftExecC2R(plan::cufftHandle, idata::CuPtr{cufftComplex},
                                 odata::CuPtr{cufftReal})::cufftResult
end

@checked function cufftExecZ2Z(plan, idata, odata, direction)
    initialize_context()
    @ccall libcufft.cufftExecZ2Z(plan::cufftHandle, idata::CuPtr{cufftDoubleComplex},
                                 odata::CuPtr{cufftDoubleComplex},
                                 direction::Cint)::cufftResult
end

@checked function cufftExecD2Z(plan, idata, odata)
    initialize_context()
    @ccall libcufft.cufftExecD2Z(plan::cufftHandle, idata::CuPtr{cufftDoubleReal},
                                 odata::CuPtr{cufftDoubleComplex})::cufftResult
end

@checked function cufftExecZ2D(plan, idata, odata)
    initialize_context()
    @ccall libcufft.cufftExecZ2D(plan::cufftHandle, idata::CuPtr{cufftDoubleComplex},
                                 odata::CuPtr{cufftDoubleReal})::cufftResult
end

@checked function cufftSetStream(plan, stream)
    initialize_context()
    @ccall libcufft.cufftSetStream(plan::cufftHandle, stream::cudaStream_t)::cufftResult
end

@checked function cufftDestroy(plan)
    initialize_context()
    @ccall libcufft.cufftDestroy(plan::cufftHandle)::cufftResult
end

@checked function cufftGetVersion(version)
    @ccall libcufft.cufftGetVersion(version::Ptr{Cint})::cufftResult
end

@checked function cufftGetProperty(type, value)
    @ccall libcufft.cufftGetProperty(type::libraryPropertyType,
                                     value::Ptr{Cint})::cufftResult
end

# Skipping MacroDefinition: CUFFTAPI __attribute__ ( ( visibility ( "default" ) ) )

const MAX_CUFFT_ERROR = 0x11

const CUFFT_FORWARD = -1

const CUFFT_INVERSE = 1

const CUFFT_COMPATIBILITY_DEFAULT = CUFFT_COMPATIBILITY_FFTW_PADDING

const MAX_SHIM_RANK = 3
