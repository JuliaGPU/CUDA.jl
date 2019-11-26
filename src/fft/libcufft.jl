# Julia wrapper for header: cufft.h
# Automatically generated using Clang.jl


function cufftPlan1d(plan, nx, type, batch)
    @check @runtime_ccall((:cufftPlan1d, libcufft[]), cufftResult,
                 (Ptr{cufftHandle}, Cint, cufftType, Cint),
                 plan, nx, type, batch)
end

function cufftPlan2d(plan, nx, ny, type)
    @check @runtime_ccall((:cufftPlan2d, libcufft[]), cufftResult,
                 (Ptr{cufftHandle}, Cint, Cint, cufftType),
                 plan, nx, ny, type)
end

function cufftPlan3d(plan, nx, ny, nz, type)
    @check @runtime_ccall((:cufftPlan3d, libcufft[]), cufftResult,
                 (Ptr{cufftHandle}, Cint, Cint, Cint, cufftType),
                 plan, nx, ny, nz, type)
end

function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
                       type, batch)
    @check @runtime_ccall((:cufftPlanMany, libcufft[]), cufftResult,
                 (Ptr{cufftHandle}, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint},
                  Cint, Cint, cufftType, Cint),
                 plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                 batch)
end

function cufftMakePlan1d(plan, nx, type, batch, workSize)
    @check @runtime_ccall((:cufftMakePlan1d, libcufft[]), cufftResult,
                 (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}),
                 plan, nx, type, batch, workSize)
end

function cufftMakePlan2d(plan, nx, ny, type, workSize)
    @check @runtime_ccall((:cufftMakePlan2d, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}),
                 plan, nx, ny, type, workSize)
end

function cufftMakePlan3d(plan, nx, ny, nz, type, workSize)
    @check @runtime_ccall((:cufftMakePlan3d, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
                 plan, nx, ny, nz, type, workSize)
end

function cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
                           type, batch, workSize)
    @check @runtime_ccall((:cufftMakePlanMany, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint,
                  Cint, cufftType, Cint, Ptr{Csize_t}),
                 plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                 batch, workSize)
end

function cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride,
                             odist, type, batch, workSize)
    @check @runtime_ccall((:cufftMakePlanMany64, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong, Clonglong,
                  Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong, Ptr{Csize_t}),
                 plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                 batch, workSize)
end

function cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride,
                            odist, type, batch, workSize)
    @check @runtime_ccall((:cufftGetSizeMany64, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong, Clonglong,
                  Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong, Ptr{Csize_t}),
                 plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                 batch, workSize)
end

function cufftEstimate1d(nx, type, batch, workSize)
    @check @runtime_ccall((:cufftEstimate1d, libcufft[]), cufftResult,
                 (Cint, cufftType, Cint, Ptr{Csize_t}),
                 nx, type, batch, workSize)
end

function cufftEstimate2d(nx, ny, type, workSize)
    @check @runtime_ccall((:cufftEstimate2d, libcufft[]), cufftResult,
                 (Cint, Cint, cufftType, Ptr{Csize_t}),
                 nx, ny, type, workSize)
end

function cufftEstimate3d(nx, ny, nz, type, workSize)
    @check @runtime_ccall((:cufftEstimate3d, libcufft[]), cufftResult,
                 (Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
                 nx, ny, nz, type, workSize)
end

function cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                           batch, workSize)
    @check @runtime_ccall((:cufftEstimateMany, libcufft[]), cufftResult,
                 (Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint,
                  cufftType, Cint, Ptr{Csize_t}),
                 rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                 workSize)
end

function cufftCreate(handle)
    @check @runtime_ccall((:cufftCreate, libcufft[]), cufftResult,
                 (Ptr{cufftHandle},),
                 handle)
end

function cufftGetSize1d(handle, nx, type, batch, workSize)
    @check @runtime_ccall((:cufftGetSize1d, libcufft[]), cufftResult,
                 (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}),
                 handle, nx, type, batch, workSize)
end

function cufftGetSize2d(handle, nx, ny, type, workSize)
    @check @runtime_ccall((:cufftGetSize2d, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}),
                 handle, nx, ny, type, workSize)
end

function cufftGetSize3d(handle, nx, ny, nz, type, workSize)
    @check @runtime_ccall((:cufftGetSize3d, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
                 handle, nx, ny, nz, type, workSize)
end

function cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride,
                          odist, type, batch, workArea)
    @check @runtime_ccall((:cufftGetSizeMany, libcufft[]), cufftResult,
                 (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint,
                  Cint, cufftType, Cint, Ptr{Csize_t}),
                 handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                 batch, workArea)
end

function cufftGetSize(handle, workSize)
    @check @runtime_ccall((:cufftGetSize, libcufft[]), cufftResult,
                 (cufftHandle, Ptr{Csize_t}),
                 handle, workSize)
end

function cufftSetWorkArea(plan, workArea)
    @check @runtime_ccall((:cufftSetWorkArea, libcufft[]), cufftResult,
                 (cufftHandle, CuPtr{Cvoid}),
                 plan, workArea)
end

function cufftSetAutoAllocation(plan, autoAllocate)
    @check @runtime_ccall((:cufftSetAutoAllocation, libcufft[]), cufftResult,
                 (cufftHandle, Cint),
                 plan, autoAllocate)
end

function cufftExecC2C(plan, idata, odata, direction)
    @check @runtime_ccall((:cufftExecC2C, libcufft[]), cufftResult,
                 (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftComplex}, Cint),
                 plan, idata, odata, direction)
end

function cufftExecR2C(plan, idata, odata)
    @check @runtime_ccall((:cufftExecR2C, libcufft[]), cufftResult,
                 (cufftHandle, CuPtr{cufftReal}, CuPtr{cufftComplex}),
                 plan, idata, odata)
end

function cufftExecC2R(plan, idata, odata)
    @check @runtime_ccall((:cufftExecC2R, libcufft[]), cufftResult,
                 (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftReal}),
                 plan, idata, odata)
end

function cufftExecZ2Z(plan, idata, odata, direction)
    @check @runtime_ccall((:cufftExecZ2Z, libcufft[]), cufftResult,
                 (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex}, Cint),
                 plan, idata, odata, direction)
end

function cufftExecD2Z(plan, idata, odata)
    @check @runtime_ccall((:cufftExecD2Z, libcufft[]), cufftResult,
                 (cufftHandle, CuPtr{cufftDoubleReal}, CuPtr{cufftDoubleComplex}),
                 plan, idata, odata)
end

function cufftExecZ2D(plan, idata, odata)
    @check @runtime_ccall((:cufftExecZ2D, libcufft[]), cufftResult,
                 (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleReal}),
                 plan, idata, odata)
end

function cufftSetStream(plan, stream)
    @check @runtime_ccall((:cufftSetStream, libcufft[]), cufftResult,
                 (cufftHandle, CUstream),
                 plan, stream)
end

function cufftDestroy(plan)
    @check @runtime_ccall((:cufftDestroy, libcufft[]), cufftResult,
                 (cufftHandle,),
                 plan)
end

function cufftGetVersion(version)
    @check @runtime_ccall((:cufftGetVersion, libcufft[]), cufftResult,
                 (Ptr{Cint},),
                 version)
end

function cufftGetProperty(type, value)
    @check @runtime_ccall((:cufftGetProperty, libcufft[]), cufftResult,
                 (libraryPropertyType, Ptr{Cint}),
                 type, value)
end
