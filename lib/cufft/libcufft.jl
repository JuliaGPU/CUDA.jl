# Julia wrapper for header: cufft.h
# Automatically generated using Clang.jl

@checked function cufftPlan1d(plan, nx, type, batch)
    initialize_api()
    ccall((:cufftPlan1d, libcufft()), cufftResult,
                   (Ptr{cufftHandle}, Cint, cufftType, Cint),
                   plan, nx, type, batch)
end

@checked function cufftPlan2d(plan, nx, ny, type)
    initialize_api()
    ccall((:cufftPlan2d, libcufft()), cufftResult,
                   (Ptr{cufftHandle}, Cint, Cint, cufftType),
                   plan, nx, ny, type)
end

@checked function cufftPlan3d(plan, nx, ny, nz, type)
    initialize_api()
    ccall((:cufftPlan3d, libcufft()), cufftResult,
                   (Ptr{cufftHandle}, Cint, Cint, Cint, cufftType),
                   plan, nx, ny, nz, type)
end

@checked function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride,
                                odist, type, batch)
    initialize_api()
    ccall((:cufftPlanMany, libcufft()), cufftResult,
                   (Ptr{cufftHandle}, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint},
                    Cint, Cint, cufftType, Cint),
                   plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                   batch)
end

@checked function cufftMakePlan1d(plan, nx, type, batch, workSize)
    initialize_api()
    ccall((:cufftMakePlan1d, libcufft()), cufftResult,
                   (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}),
                   plan, nx, type, batch, workSize)
end

@checked function cufftMakePlan2d(plan, nx, ny, type, workSize)
    initialize_api()
    ccall((:cufftMakePlan2d, libcufft()), cufftResult,
                   (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}),
                   plan, nx, ny, type, workSize)
end

@checked function cufftMakePlan3d(plan, nx, ny, nz, type, workSize)
    initialize_api()
    ccall((:cufftMakePlan3d, libcufft()), cufftResult,
                   (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
                   plan, nx, ny, nz, type, workSize)
end

@checked function cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed,
                                    ostride, odist, type, batch, workSize)
    initialize_api()
    ccall((:cufftMakePlanMany, libcufft()), cufftResult,
                   (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint,
                    Cint, cufftType, Cint, Ptr{Csize_t}),
                   plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                   batch, workSize)
end

@checked function cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed,
                                      ostride, odist, type, batch, workSize)
    initialize_api()
    ccall((:cufftMakePlanMany64, libcufft()), cufftResult,
                   (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong,
                    Clonglong, Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong,
                    Ptr{Csize_t}),
                   plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                   batch, workSize)
end

@checked function cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed,
                                     ostride, odist, type, batch, workSize)
    initialize_api()
    ccall((:cufftGetSizeMany64, libcufft()), cufftResult,
                   (cufftHandle, Cint, Ptr{Clonglong}, Ptr{Clonglong}, Clonglong,
                    Clonglong, Ptr{Clonglong}, Clonglong, Clonglong, cufftType, Clonglong,
                    Ptr{Csize_t}),
                   plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                   batch, workSize)
end

@checked function cufftEstimate1d(nx, type, batch, workSize)
    initialize_api()
    ccall((:cufftEstimate1d, libcufft()), cufftResult,
                   (Cint, cufftType, Cint, Ptr{Csize_t}),
                   nx, type, batch, workSize)
end

@checked function cufftEstimate2d(nx, ny, type, workSize)
    initialize_api()
    ccall((:cufftEstimate2d, libcufft()), cufftResult,
                   (Cint, Cint, cufftType, Ptr{Csize_t}),
                   nx, ny, type, workSize)
end

@checked function cufftEstimate3d(nx, ny, nz, type, workSize)
    initialize_api()
    ccall((:cufftEstimate3d, libcufft()), cufftResult,
                   (Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
                   nx, ny, nz, type, workSize)
end

@checked function cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                                    odist, type, batch, workSize)
    initialize_api()
    ccall((:cufftEstimateMany, libcufft()), cufftResult,
                   (Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Cint,
                    cufftType, Cint, Ptr{Csize_t}),
                   rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                   workSize)
end

@checked function cufftCreate(handle)
    initialize_api()
    ccall((:cufftCreate, libcufft()), cufftResult,
                   (Ptr{cufftHandle},),
                   handle)
end

@checked function cufftGetSize1d(handle, nx, type, batch, workSize)
    initialize_api()
    ccall((:cufftGetSize1d, libcufft()), cufftResult,
                   (cufftHandle, Cint, cufftType, Cint, Ptr{Csize_t}),
                   handle, nx, type, batch, workSize)
end

@checked function cufftGetSize2d(handle, nx, ny, type, workSize)
    initialize_api()
    ccall((:cufftGetSize2d, libcufft()), cufftResult,
                   (cufftHandle, Cint, Cint, cufftType, Ptr{Csize_t}),
                   handle, nx, ny, type, workSize)
end

@checked function cufftGetSize3d(handle, nx, ny, nz, type, workSize)
    initialize_api()
    ccall((:cufftGetSize3d, libcufft()), cufftResult,
                   (cufftHandle, Cint, Cint, Cint, cufftType, Ptr{Csize_t}),
                   handle, nx, ny, nz, type, workSize)
end

@checked function cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed,
                                   ostride, odist, type, batch, workArea)
    initialize_api()
    ccall((:cufftGetSizeMany, libcufft()), cufftResult,
                   (cufftHandle, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint,
                    Cint, cufftType, Cint, Ptr{Csize_t}),
                   handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                   batch, workArea)
end

@checked function cufftGetSize(handle, workSize)
    initialize_api()
    ccall((:cufftGetSize, libcufft()), cufftResult,
                   (cufftHandle, Ptr{Csize_t}),
                   handle, workSize)
end

@checked function cufftSetWorkArea(plan, workArea)
    initialize_api()
    ccall((:cufftSetWorkArea, libcufft()), cufftResult,
                   (cufftHandle, CuPtr{Cvoid}),
                   plan, workArea)
end

@checked function cufftSetAutoAllocation(plan, autoAllocate)
    initialize_api()
    ccall((:cufftSetAutoAllocation, libcufft()), cufftResult,
                   (cufftHandle, Cint),
                   plan, autoAllocate)
end

@checked function cufftExecC2C(plan, idata, odata, direction)
    initialize_api()
    ccall((:cufftExecC2C, libcufft()), cufftResult,
                   (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftComplex}, Cint),
                   plan, idata, odata, direction)
end

@checked function cufftExecR2C(plan, idata, odata)
    initialize_api()
    ccall((:cufftExecR2C, libcufft()), cufftResult,
                   (cufftHandle, CuPtr{cufftReal}, CuPtr{cufftComplex}),
                   plan, idata, odata)
end

@checked function cufftExecC2R(plan, idata, odata)
    initialize_api()
    ccall((:cufftExecC2R, libcufft()), cufftResult,
                   (cufftHandle, CuPtr{cufftComplex}, CuPtr{cufftReal}),
                   plan, idata, odata)
end

@checked function cufftExecZ2Z(plan, idata, odata, direction)
    initialize_api()
    ccall((:cufftExecZ2Z, libcufft()), cufftResult,
                   (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex},
                    Cint),
                   plan, idata, odata, direction)
end

@checked function cufftExecD2Z(plan, idata, odata)
    initialize_api()
    ccall((:cufftExecD2Z, libcufft()), cufftResult,
                   (cufftHandle, CuPtr{cufftDoubleReal}, CuPtr{cufftDoubleComplex}),
                   plan, idata, odata)
end

@checked function cufftExecZ2D(plan, idata, odata)
    initialize_api()
    ccall((:cufftExecZ2D, libcufft()), cufftResult,
                   (cufftHandle, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleReal}),
                   plan, idata, odata)
end

@checked function cufftSetStream(plan, stream)
    initialize_api()
    ccall((:cufftSetStream, libcufft()), cufftResult,
                   (cufftHandle, CUstream),
                   plan, stream)
end

@checked function cufftDestroy(plan)
    initialize_api()
    ccall((:cufftDestroy, libcufft()), cufftResult,
                   (cufftHandle,),
                   plan)
end

@checked function cufftGetVersion(version)
    ccall((:cufftGetVersion, libcufft()), cufftResult,
                   (Ptr{Cint},),
                   version)
end

@checked function cufftGetProperty(type, value)
    ccall((:cufftGetProperty, libcufft()), cufftResult,
                   (libraryPropertyType, Ptr{Cint}),
                   type, value)
end
