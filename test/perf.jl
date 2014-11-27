using CUDA, Base.Test

include("perfutil.jl")


# set-up

dev = CuDevice(0)
ctx = CuContext(dev)

initialize_codegen(ctx, dev)


# measurement 1: @cuda time (purely the staged function)

export dummy

@target ptx dummy() = return nothing

@timeit begin
        @eval @cuda (0, 0) dummy()
    end "cuda_lowered" "lowered @cuda"


# measurement 2: compilation time

i = 0
@timeit_init begin
        @eval @cuda (0, 0) $fname()
    end begin
        # initialization
        i += 1
        fname = symbol("dummy_$i")
        @eval @target ptx $fname() = return nothing
    end "cuda_unlowered" "unlowered @cuda + compilation"
