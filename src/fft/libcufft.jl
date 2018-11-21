cufftGetVersion() = ccall((:cufftGetVersion,libcufft), Cint, ())

function cufftGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  @check ccall((:cufftGetProperty, libcufft),
               cufftStatus_t,
               (Cint, Ptr{Cint}),
               property, value_ref)
  value_ref[]
end
