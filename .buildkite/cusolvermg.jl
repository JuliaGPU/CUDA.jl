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
        # what the test does on this agent (two GPUs visible)
        ("one device (default), all GPUs visible", Dict()),
        ("one device (default), only GPU 0 visible", Dict("CUDA_VISIBLE_DEVICES" => "0")),
        ("one device (default), only GPU 1 visible", Dict("CUDA_VISIBLE_DEVICES" => "1")),
        # what the test does on single-GPU agents
        ("same device twice, all GPUs visible", Dict("MG_DEVICES" => "same")),
        ("same device twice, only GPU 0 visible",
         Dict("MG_DEVICES" => "same", "CUDA_VISIBLE_DEVICES" => "0")),
        ("both GPUs", Dict("MG_DEVICES" => "all")),
        ("one device (default), cuSOLVER logging",
         Dict("CUSOLVER_LOG_LEVEL" => "5", "CUSOLVER_LOG_MASK" => "31")),
    ]
    for (name, env) in variants
        println("\n--- $name")
        run(ignorestatus(addenv(`$julia child`, env)))
    end

    println("\n--- one device (default), under compute-sanitizer (reporting API errors)")
    cs = sanitizer()
    run(ignorestatus(`$cs --tool=memcheck --launch-timeout=0 --report-api-errors=all $julia child`))
end

if isempty(ARGS)
    main()
else
    run_operations()
end
