# debug functionality

"""
    enable_timings()

Enable the recording of debug timings.
"""
enable_timings() = (TimerOutputs.enable_debug_timings(CUDA); return)
disable_timings() = (TimerOutputs.disable_debug_timings(CUDA); return)
