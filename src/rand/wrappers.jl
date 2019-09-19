function curandGetVersion()
    ver = Ref{Cint}()
    curandGetVersion(ver)
    return ver[]
end

function curandGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end
