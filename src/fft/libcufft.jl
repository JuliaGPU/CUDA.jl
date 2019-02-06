# low-level wrappers of the CUFFT library

cufftGetVersion() = ccall((:cufftGetVersion,libcufft), Cint, ())

function cufftGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  @check ccall((:cufftGetProperty, libcufft), cufftStatus_t,
               (Cint, Ptr{Cint}),
               property, value_ref)
  value_ref[]
end

cufftDestroy(plan) = ccall((:cufftDestroy,libcufft), Nothing, (cufftHandle_t,), plan)

function cufftPlan1d(plan, nx, type, batch)
    @check ccall((:cufftPlan1d,libcufft),cufftStatus_t,
                 (Ptr{cufftHandle_t}, Cint, cufftType, Cint),
                 plan, nx, type, batch)
end

function cufftPlan2d(plan, nx, ny, type)
    @check ccall((:cufftPlan2d,libcufft),cufftStatus_t,
                 (Ptr{cufftHandle_t}, Cint, Cint, cufftType),
                 plan, nx, ny, type)
end

function cufftPlan3d(plan, nx, ny, nz, type)
    @check ccall((:cufftPlan3d,libcufft),cufftStatus_t,
                 (Ptr{cufftHandle_t}, Cint, Cint, Cint, cufftType),
                 plan, nx, ny, nz, type)
end

function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)
    @check ccall((:cufftPlanMany,libcufft),cufftStatus_t,
                 (Ptr{cufftHandle_t}, Cint, Ptr{Cint},
                  Ptr{Cint}, Cint, Cint,
                  Ptr{Cint}, Cint, Cint,
                  cufftType, Cint),
                 plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)
end

function cufftExecC2C(plan, idata, odata, direction)
    @check ccall((:cufftExecC2C,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftComplex}, CuPtr{cufftComplex}, Cint),
                 plan, idata, odata, direction)
end

function cufftExecC2R(plan, idata, odata)
    @check ccall((:cufftExecC2R,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftComplex}, CuPtr{cufftComplex}),
                 plan, idata, odata)
end

function cufftExecR2C(plan, idata, odata)
    @check ccall((:cufftExecR2C,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftReal}, CuPtr{cufftComplex}),
                 plan, idata, odata)
end

function cufftExecZ2Z(plan, idata, odata, direction)
    @check ccall((:cufftExecZ2Z,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex},
                  Cint),
                 plan, idata, odata, direction)
end

function cufftExecZ2D(plan, idata, odata)
    @check ccall((:cufftExecZ2D,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex}),
                 plan, idata, odata)
end

function cufftExecD2Z(plan, idata, odata)
    @check ccall((:cufftExecD2Z,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleReal}, CuPtr{cufftDoubleComplex}),
                 plan, idata, odata)
end
