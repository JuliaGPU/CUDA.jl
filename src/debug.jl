# debug functionality

"""
    enable_timings()

Enable the recording of debug timings.
"""
enable_timings() = (TimerOutputs.enable_debug_timings(CUDA); return)
disable_timings() = (TimerOutputs.disable_debug_timings(CUDA); return)

isdebug(group, mod=CUDA) =
    Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug, group, mod) !== nothing
