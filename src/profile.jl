export @profile

"""
    @profile

`@profile <expression>` runs your expression, while activating the CUDA profiler upon first
kernel launch.
"""
macro profile(ex)
    quote
        Profile.enabled[] = true
        try
            $(esc(ex))
        finally
            Profile.started[] && CUDAdrv.Profile.stop()
            Profile.enabled[] = false
            Profile.started[] = false
        end
    end
end


module Profile

using CUDAdrv

const enabled = Ref(false)
const started = Ref(false)

macro launch(ex)
    quote
        if enabled[] && !started[]
            started[] = true
            CUDAdrv.Profile.start()
        end

        $(esc(ex))
    end
end

end
