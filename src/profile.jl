# Profiler interface

export
    @cuprofile

@enum(CUoutput_mode, CU_OUT_CSV = Cint(0),
                     CU_OUT_KEY_VALUE_PAIR)

start_profiler() = @cucall(:cuProfilerStart, (Ptr{Void},), C_NULL)
stop_profiler() = @cucall(:cuProfilerStop, (Ptr{Void},), C_NULL)

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
