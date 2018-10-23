# Deprecated functionality

macro profile(ex)
    Base.depwarn("`CUDAnative.@profile` is deprecated, use `CUDAdrv.@profile` instead", :profile)
    quote
        CUDAdrv.@profile begin
            $(esc(ex))
        end
    end 
end

@deprecate cudaconvert(x) adapt(CUDAnative.Adaptor, x)
