# @profile infrastructure (GPU-dependent, can't execute during precompilation)
precompile(Tuple{typeof(Profile.detect_cupti)})
precompile(Tuple{typeof(Profile.profile_internally), Function})
precompile(Tuple{typeof(Profile.capture), CUPTI.ActivityConfig})

using PrecompileTools

@compile_workload begin
    # exercise the @profile display path with a dummy result (no GPU needed).
    # the show method expects at least two cuCtxSynchronize entries in the host trace
    # to delimit the profiled region, and at least one event between them.
    dummy = Profile.ProfileResults(;
        host = (
            id      = Int[1, 2, 3, 4],
            start   = Float64[0.0, 0.001, 0.002, 0.010],
            stop    = Float64[0.001, 0.002, 0.009, 0.011],
            name    = String["cuCtxSynchronize", "cuCtxSynchronize",
                             "cuLaunchKernel", "cuCtxSynchronize"],
            tid     = Int[1, 1, 1, 1],
        ),
        device = (
            id      = Int[3],
            start   = Float64[0.003],
            stop    = Float64[0.008],
            name    = String["kernel"],
            device  = Int[0],
            context = Int[1],
            stream  = Int[1],
            grid            = Union{Missing,CUDACore.CuDim3}[CUDACore.CuDim3(1,1,1)],
            block           = Union{Missing,CUDACore.CuDim3}[CUDACore.CuDim3(1,1,1)],
            registers       = Union{Missing,Int64}[32],
            shared_mem      = Union{Missing,@NamedTuple{static::Int64,dynamic::Int64}}[(static=0,dynamic=0)],
            local_mem       = Union{Missing,@NamedTuple{thread::Int64,total::Int64}}[(thread=0,total=0)],
            size            = Union{Missing,Int64}[missing],
        ),
        nvtx = (
            id      = Int[],
            start   = Float64[],
            type    = Symbol[],
            tid     = Int[],
            name    = Union{Missing,String}[],
            domain  = Union{Missing,String}[],
        ),
    )
    show(devnull, dummy)
end
