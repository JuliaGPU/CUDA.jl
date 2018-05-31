# logging

using Logging

# FIXME: replace with an additional log level
macro trace(ex...)
    esc(:(@debug $(ex...)))
end

# fatal versions to `@error`, including a safe version for compile-time errors
# (to be used instead of `error`, which _emits_ an error)
macro fatal(ex...)
    esc(quote
        @error $(ex...)
        error("Fatal error occurred")
    end)
end

# define safe loggers for use in generated functions (where task switches are not allowed)
for level in [:trace, :debug, :info, :warn, :error, :fatal]
    @eval begin
        macro $(Symbol("safe_$level"))(ex...)
            macrocall = :(@placeholder $(ex...))
            # NOTE: `@placeholder` in order to avoid hard-coding @__LINE__ etc
            macrocall.args[1] = Symbol($"@$level")
            quote
                old_logger = global_logger()
                global_logger(Logging.ConsoleLogger(Core.stderr, old_logger.min_level))
                ret = $(esc(macrocall))
                global_logger(old_logger)
                ret
            end
        end
    end
end
