# Per-test helpers, `include`d into each test's sandbox module. Kept separate
# from `setup.jl` (which runs once per worker and defines `CUDATestRecord`)
# so subpackage tests' own setup.jl can redefine `testf` without fighting a
# `using`-imported binding.

using CUDA, CUDACore, GPUArrays
using CUDA: i32
using Adapt
using ..Main: TestSuite, can_use_cupti, sanitize

testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))

                    # NOTE: CUDA requires a 'proper' sync to flush its printf buffer
                    synchronize(context())
                end
            end
            ret, read(fname, String)
        end
    end
end

# Run some code on-device
macro on_device(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    @gensym kernel
    esc(quote
        let
            function $kernel()
                $code
                return
            end

            CUDA.@sync @cuda $(kwargs...) $kernel()
        end
    end)
end

# helper function for sinking a value to prevent the callee from getting optimized away
@inline sink(i::Int32) =
    Base.llvmcall("""%slot = alloca i32
                     store volatile i32 %0, i32* %slot
                     %value = load volatile i32, i32* %slot
                     ret i32 %value""", Int32, Tuple{Int32}, i)
@inline sink(i::Int64) =
    Base.llvmcall("""%slot = alloca i64
                     store volatile i64 %0, i64* %slot
                     %value = load volatile i64, i64* %slot
                     ret i64 %value""", Int64, Tuple{Int64}, i)

function julia_exec(args::Cmd, env...)
    # FIXME: this doesn't work when the compute mode is set to exclusive
    cmd = Base.julia_cmd()
    cmd = `$cmd --project=$(Base.active_project()) --color=no $args`

    out = Pipe()
    err = Pipe()
    proc = run(pipeline(addenv(cmd, env...), stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)
    wait(proc)
    proc, read(out, String), read(err, String)
end
