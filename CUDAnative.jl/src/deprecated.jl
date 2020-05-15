# Deprecated functionality

macro profile(ex)
    Base.depwarn("`CUDAnative.@profile` is deprecated, use `CUDAdrv.@profile` instead", :profile)
    quote
        CUDAdrv.@profile begin
            $(esc(ex))
        end
    end
end

@deprecate nearest_warpsize(dev::CuDevice, threads::Integer) nextwarp(dev, threads)
