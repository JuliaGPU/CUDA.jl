# debug functionality

isdebug(group, mod=CUDACore) =
    Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug, group, mod) !== nothing
