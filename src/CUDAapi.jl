__precompile__()

module CUDAapi

using Libdl
using Logging

# FIXME: replace with an additional log level when we depend on 0.7+
macro trace(ex...)
    esc(:(@debug $(ex...)))
end

include("util.jl")
include("compatibility.jl")
include("discovery.jl")

function __init__()
    DEBUG = parse(Bool, get(ENV, "DEBUG", "false"))
    if DEBUG
        global_logger(ConsoleLogger(global_logger().stream, Logging.Debug))
    end
end

end
