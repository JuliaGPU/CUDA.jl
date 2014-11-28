using ArgParse

using CUDA, Base.Test

s = ArgParseSettings()
@add_arg_table s begin
    "--performance", "-p"
        help = "Perform performance measurements"
        action = :store_true
    "--server"
        help = "Codespeed server(s) to submit performance measurements to"
        nargs = 1
        arg_type = String
        action = :append_arg
end
opts = parse_args(ARGS, s)
if length(opts["server"]) > 0 && !opts["performance"]
    error("Cannot submit to Codespeed without enabling performance measurements")
end

include("perfutil.jl")

@test devcount() > 0
include("core.jl")

dev = CuDevice(0)
if capability(dev) < v"2.0"
    warn("native execution not supported on SM < 2.0")
else
    include("native.jl")
end
