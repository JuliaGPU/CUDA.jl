# Profiler interface

export
    @cuprofile

@enum(CUoutput_mode, CSV            = 0x00,
                     KEY_VALUE_PAIR = 0x01)

start_profiler() = @apicall(:cuProfilerStart, (Ptr{Void},), C_NULL)
stop_profiler() = @apicall(:cuProfilerStop, (Ptr{Void},), C_NULL)

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
