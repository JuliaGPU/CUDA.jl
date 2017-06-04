export @profile

"""
    @profile ex

Runs your expression `ex` while activating the CUDA profiler upon first kernel launch. This
makes it easier to profile accurately, without the overhead of initial compilation, memory
transfers, ...

Note that this API is used to programmatically control the profiling granularity by allowing
profiling to be done only on selective pieces of code. It does not perform any profiling on
itself, you need external tools for that.
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
