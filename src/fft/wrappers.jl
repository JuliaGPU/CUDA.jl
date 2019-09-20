# wrappers of the low-level CUBLAS functionality

function cufftGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(value_ref)
  value_ref[]
end
