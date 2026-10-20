# Diagnose cuSOLVERMg returning CUSOLVER_STATUS_INTERNAL_ERROR on the MultiGPU agent
# (2x Quadro RTX 5000), see JuliaGPU/CUDA.jl#3358. Without arguments, runs every variant
# below in a fresh process; with an argument, runs the operations in this process.

using CUDACore, cuSOLVER, LinearAlgebra, Pkg

const operations = ["syevd_V", "syevd_N", "potrf", "getrf"]

function run_operations()
    println("devices: ", collect(CUDACore.devices()), ", runtime ", CUDACore.runtime_version(),
            ", cuSOLVER ", cuSOLVER.version())
    devs = collect(CUDACore.devices())
    if length(devs) > 1
        println("can_access_peer: ", CUDACore.can_access_peer(devs[1], devs[2]))
    end
    # state left behind by tests that ran earlier in the same worker (core/array's
    # "multigpu" testset creates a context on the second device and copies across devices)
    pre = get(ENV, "MG_PRE", "none")
    if pre != "none" && length(devs) > 1
        d0, d1 = devs
        if pre == "touch"
            device!(d1); CuArray([1]); synchronize(); device!(d0)
        elseif pre == "dev1"
            device!(d1)
        elseif pre == "copy"
            data = rand(5, 5)
            device!(d0); a = CuArray(data)
            device!(d1); b = similar(a); copyto!(b, a); synchronize()
            device!(d0); copyto!(a, b); synchronize()
            println("cross-device copy: ", Array(a) == Array(b) == data)
            u = CuArray{Float64, 2, CUDACore.UnifiedMemory}(undef, 5, 5)
            copyto!(u, a); synchronize()
            println("unified copy: ", Array(u) == data)
        elseif pre == "peer"
            println("peer: ", CUDACore.maybe_enable_peer_access(d0, d1), " ",
                    CUDACore.maybe_enable_peer_access(d1, d0))
        end
        println("current device after MG_PRE=$pre: ", device())
    end
    # the test only doubles the device when a single GPU is visible; on the MultiGPU agent
    # it leaves cuSOLVERMg at its default, a one-device grid on the first GPU.
    mg = get(ENV, "MG_DEVICES", "default")
    if mg == "same"
        cuSOLVER.devices!([device(), device()])
    elseif mg == "all"
        cuSOLVER.devices!(devs)
    end
    println("cuSOLVERMg devices: ", cuSOLVER.devices())
    m = 256
    for op in operations, elty in (Float32, ComplexF64)
        A = rand(elty, m, m)
        try
            if op == "syevd_V"
                A += A'
                W, V = cuSOLVER.mg_syevd!('V', 'L', copy(A))
                ok = W ≈ eigvals(Hermitian(A))
            elseif op == "syevd_N"
                A += A'
                W = cuSOLVER.mg_syevd!('N', 'L', copy(A))
                ok = W ≈ eigvals(Hermitian(A))
            elseif op == "potrf"
                A = A * A' + m * I
                cuSOLVER.mg_potrf!('L', copy(A))
                ok = true
            elseif op == "getrf"
                cuSOLVER.mg_getrf!(copy(A))
                ok = true
            end
            println("  $op $elty: ", ok ? "ok" : "WRONG RESULT")
        catch err
            println("  $op $elty: ", sprint(showerror, err))
        end
    end
end

function sanitizer()
    project = Base.active_project()
    Pkg.activate(; temp = true, io = devnull)
    Pkg.add("CUDA_SDK_jll"; io = devnull)
    @eval using CUDA_SDK_jll
    Pkg.activate(project; io = devnull)
    sdk = Base.invokelatest(getglobal, @__MODULE__, :CUDA_SDK_jll)
    joinpath(Base.invokelatest(getproperty, sdk, :artifact_dir),
             "cuda", "compute-sanitizer", "compute-sanitizer")
end

function main()
    run(ignorestatus(`nvidia-smi`))
    run(ignorestatus(`nvidia-smi topo -m`))
    run(ignorestatus(`nvidia-smi topo -p2p r`))

    julia = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__)`
    variants = [
        # what the test does on this agent (two GPUs visible); passed in build 8234
        ("one device (default), fresh process", Dict()),
        # state left behind by core/array, which precedes the test in the same worker
        ("one device after creating a context on GPU 1", Dict("MG_PRE" => "touch")),
        ("one device with GPU 1 current", Dict("MG_PRE" => "dev1")),
        ("one device after cross-device and unified copies", Dict("MG_PRE" => "copy")),
        ("one device with peer access enabled", Dict("MG_PRE" => "peer")),
        ("both GPUs, cuSOLVER logging",
         Dict("MG_DEVICES" => "all", "CUSOLVER_LOG_LEVEL" => "5", "CUSOLVER_LOG_MASK" => "31")),
        # the cuSolverMG docs (3.1.8) leave enabling peer access to the user; our wrappers
        # never do, so a grid on two real GPUs may need it
        ("both GPUs with peer access enabled", Dict("MG_DEVICES" => "all", "MG_PRE" => "peer")),
    ]
    for (name, env) in variants
        println("\n--- $name")
        run(ignorestatus(addenv(`$julia child`, env)))
    end

    cs = sanitizer()
    for (name, env) in [
            ("one device after creating a context on GPU 1", Dict("MG_PRE" => "touch")),
            ("both GPUs", Dict("MG_DEVICES" => "all")),
            ("both GPUs with peer access enabled", Dict("MG_DEVICES" => "all", "MG_PRE" => "peer")),
        ]
        println("\n--- $name, under compute-sanitizer (reporting API errors)")
        run(ignorestatus(addenv(`$cs --tool=memcheck --launch-timeout=0 --report-api-errors=all $julia child`, env)))
    end
end

if isempty(ARGS)
    main()
else
    run_operations()
end
