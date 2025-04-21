# debug functionality

function isdebug(group, mod=CUDA)
    level = Base.CoreLogging.Debug
    logger = Base.CoreLogging.current_logger_for_env(level, group, mod)
    # TODO: Which id to choose here instead of 0?
    logger !== nothing && Base.CoreLogging.shouldlog(logger, level, mod, group, 0)
end
