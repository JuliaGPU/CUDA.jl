# Profiler interface

export
    @cuprofile, init_profiler

initialized = false

start_profiler() = @cucall(:cuProfilerStart, (Ptr{Void},), 0)
stop_profiler() = @cucall(:cuProfilerStop, (Ptr{Void},), 0)

macro cuprofile(ex)
    quote
        start_profiler()
        try
            $(esc(ex))
        finally
            stop_profiler()
        end
    end
end
