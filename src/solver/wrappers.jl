# low-level wrappers of the CUSOLVER library

function cusolverDnCreate()
  handle = Ref{cusolverDnHandle_t}()
  cusolverDnCreate(handle)
  return handle[]
end

function cusolverSpCreate()
  handle = Ref{cusolverSpHandle_t}()
  cusolverSpCreate(handle)
  return handle[]
end

function cusolverGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  cusolverGetProperty(property, value_ref)
  value_ref[]
end
