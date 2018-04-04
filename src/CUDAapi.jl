__precompile__()

module CUDAapi

using Compat
if VERSION >= v"0.7-"
    using Libdl
    using Logging
else
    using MicroLogging
end

include("util.jl")
include("compatibility.jl")
include("discovery.jl")

function __init__()
    DEBUG = parse(Bool, get(ENV, "DEBUG", "false"))
    if DEBUG
        if VERSION >= v"0.7-"
            global_logger(ConsoleLogger(global_logger().stream, Logging.Debug))
        else
            configure_logging(min_level=:debug)
        end
    end
end

end
