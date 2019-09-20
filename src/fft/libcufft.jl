# Julia wrapper for header: cufft.h
# Automatically generated using Clang.jl


function cufftPlan1d(plan, nx, type, batch)
    ccall((:cufftPlan1d, libcufft), cufftResult,
          (Ptr{cufftHandle}, Cint, cufftType, Cint),
          plan, nx, type, batch)
end

function cufftPlan2d(plan, nx, ny, type)
    ccall((:cufftPlan2d, libcufft), cufftResult,
          (Ptr{cufftHandle}, Cint, Cint, cufftType),
          plan, nx, ny, type)
end

function cufftPlan3d(plan, nx, ny, nz, type)
    ccall((:cufftPlan3d, libcufft), cufftResult,
          (Ptr{cufftHandle}, Cint, Cint, Cint, cufftType),
          plan, nx, ny, nz, type)
end

function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
                       type, batch)
    ccall((:cufftPlanMany, libcufft), cufftResult,
          (Ptr{cufftHandle}, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint,
           Cint, cufftType, Cint),
          plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)
end

function cufftMakePlan1d(plan, nx, type, batch, workSize)
    ccall((:cufftMakePlan1d, libcufft), cufftResult,
          (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}),
          plan, nx, type, batch, workSize)
end

function cufftMakePlan2d(plan, nx, ny, type, workSize)
    ccall((:cufftMakePlan2d, libcufft), cufftResult,
          (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}),
          plan, nx, ny, type, workSize)
end

function cufftMakePlan3d(plan, nx, ny, nz, type, workSize)
    ccall((:cufftMakePlan3d, libcufft), cufftResult,
          (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
          plan, nx, ny, nz, type, workSize)
end

function cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
                           type, batch, workSize)
    ccall((:cufftMakePlanMany, libcufft), cufftResult,
          (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint,
           cufftType, Cint, Ptr{Csize_t}),
          plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
          workSize)
end

function cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride,
                             odist, type, batch, workSize)
    ccall((:cufftMakePlanMany64, libcufft), cufftResult,
          (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong, Clonglong,
           Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong, Ptr{Csize_t}),
          plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
          workSize)
end

function cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride,
                            odist, type, batch, workSize)
    ccall((:cufftGetSizeMany64, libcufft), cufftResult,
          (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong, Clonglong,
           Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong, Ptr{Csize_t}),
          plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
          workSize)
end

function cufftEstimate1d(nx, type, batch, workSize)
    ccall((:cufftEstimate1d, libcufft), cufftResult,
          (Cint, cufftType, Cint, Ptr{Csize_t}),
          nx, type, batch, workSize)
end

function cufftEstimate2d(nx, ny, type, workSize)
    ccall((:cufftEstimate2d, libcufft), cufftResult,
          (Cint, Cint, cufftType, Ptr{Csize_t}),
          nx, ny, type, workSize)
end

function cufftEstimate3d(nx, ny, nz, type, workSize)
    ccall((:cufftEstimate3d, libcufft), cufftResult,
          (Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
          nx, ny, nz, type, workSize)
end

function cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                           batch, workSize)
    ccall((:cufftEstimateMany, libcufft), cufftResult,
          (Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint, cufftType, Cint,
           Ptr{Csize_t}),
          rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)
end

function cufftCreate(handle)
    ccall((:cufftCreate, libcufft), cufftResult,
          (Ptr{cufftHandle},),
          handle)
end

function cufftGetSize1d(handle, nx, type, batch, workSize)
    ccall((:cufftGetSize1d, libcufft), cufftResult,
          (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}),
          handle, nx, type, batch, workSize)
end

function cufftGetSize2d(handle, nx, ny, type, workSize)
    ccall((:cufftGetSize2d, libcufft), cufftResult,
          (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}),
          handle, nx, ny, type, workSize)
end

function cufftGetSize3d(handle, nx, ny, nz, type, workSize)
    ccall((:cufftGetSize3d, libcufft), cufftResult,
          (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
          handle, nx, ny, nz, type, workSize)
end

function cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride,
                          odist, type, batch, workArea)
    ccall((:cufftGetSizeMany, libcufft), cufftResult,
          (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint,
           cufftType, Cint, Ptr{Csize_t}),
          handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
          workArea)
end

function cufftGetSize(handle, workSize)
    ccall((:cufftGetSize, libcufft), cufftResult,
          (cufftHandle, Ptr{Csize_t}),
          handle, workSize)
end

function cufftSetWorkArea(plan, workArea)
    ccall((:cufftSetWorkArea, libcufft), cufftResult,
          (cufftHandle, CuPtr{Cvoid}),
          plan, workArea)
end

function cufftSetAutoAllocation(plan, autoAllocate)
    ccall((:cufftSetAutoAllocation, libcufft), cufftResult,
          (cufftHandle, Cint),
          plan, autoAllocate)
end

function cufftExecC2C(plan, idata, odata, direction)
    ccall((:cufftExecC2C, libcufft), cufftResult,
          (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftComplex}, Cint),
          plan, idata, odata, direction)
end

function cufftExecR2C(plan, idata, odata)
    ccall((:cufftExecR2C, libcufft), cufftResult,
          (cufftHandle, CuPtr{cufftReal}, CuPtr{cufftComplex}),
          plan, idata, odata)
end

function cufftExecC2R(plan, idata, odata)
    ccall((:cufftExecC2R, libcufft), cufftResult,
          (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftReal}),
          plan, idata, odata)
end

function cufftExecZ2Z(plan, idata, odata, direction)
    ccall((:cufftExecZ2Z, libcufft), cufftResult,
          (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex}, Cint),
          plan, idata, odata, direction)
end

function cufftExecD2Z(plan, idata, odata)
    ccall((:cufftExecD2Z, libcufft), cufftResult,
          (cufftHandle, CuPtr{cufftDoubleReal}, CuPtr{cufftDoubleComplex}),
          plan, idata, odata)
end

function cufftExecZ2D(plan, idata, odata)
    ccall((:cufftExecZ2D, libcufft), cufftResult,
          (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleReal}),
          plan, idata, odata)
end

function cufftSetStream(plan, stream)
    ccall((:cufftSetStream, libcufft), cufftResult,
          (cufftHandle, CuStream_t),
          plan, stream)
end

function cufftDestroy(plan)
    ccall((:cufftDestroy, libcufft), cufftResult,
          (cufftHandle,),
          plan)
end

function cufftGetVersion(version)
    ccall((:cufftGetVersion, libcufft), cufftResult,
          (Ptr{Cint},),
          version)
end

function cufftGetProperty(type, value)
    ccall((:cufftGetProperty, libcufft), cufftResult,
          (libraryPropertyType, Ptr{Cint}),
          type, value)
end
