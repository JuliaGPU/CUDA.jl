using CUDA
using BenchmarkTools
using LLVM
using LLVM.Interop
using Core: LLVMPtr

function hash_64_64(n::Int64)
    a::Int64 = n
    a = ~a + a << 21
    a =  a ⊻ a >> 24
    a =  a + a << 3 + a << 8
    a =  a ⊻ a >> 14
    a =  a + a << 2 + a << 4
    a =  a ⊻ a >> 28
    a =  a + a << 31
    return a
end


function hashtimes(x, j)
    i = 0
    while i < j
        i += 1
        x = hash_64_64(x)
    end
    x
end


function is_important(x)
    return (hashtimes(x, 100) % 100) == 0
end



mutable struct Sampler
    at::Int64
    total::Int64
    numbers::Vector{Int64}
end

Sampler(total::Int64) = Sampler(0, total, Int64[])
reset!(sampler::Sampler) = (sampler.at=0; empty!(sampler.numbers))

function get_samples(m, sampler)
    sampler.at == sampler.total && return (-1, -1)

    current = sampler.at
    if current + m > sampler.total
        sampler.at = sampler.total
        return (sampler.total - current, current)
    end

    sampler.at += m
    (m, current)
end


function notify_important(x, sampler)
    push!(sampler.numbers, x)
    return
end


function main_kernel_blocking(m::Int64, sampler)
    (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) get_samples(m, sampler)
    while i != -1
        j = 0
        while j < i
            j += 1
            if is_important(v)
                @CUDA.cpu blocking=true types=(Nothing, Int64, CUDA.HostRef,) notify_important(v, sampler)
            end
            sync_warp()

            v += 1
        end

        (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) get_samples(m, sampler)
    end
    return
end

function main_kernel_non_blocking(m::Int64, sampler)
    (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) get_samples(m, sampler)
    while i != -1
        j = 0
        while j < i
            j += 1
            if is_important(v)
                @CUDA.cpu blocking=false types=(Nothing, Int64, CUDA.HostRef,) notify_important(v, sampler)
            end
            sync_warp()

            v += 1
        end

        (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) get_samples(m, sampler)
    end
    return
end

gather((m1, sampler1), (m2, sampler2)) = ((m1, m2), (m1+m2, sampler1)) # ask double as much, same sampler
function scatter((m1, m2), (count, current))
    count < 0 && return ((-1, -1), (-1, -1))
    c1 = div(count, 2)
    c2 = count - c1
    return ((c1, current), (c2, current + c1))
end

function kernel_gathered_blocking(m::Int64, sampler)
    (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) gran_gather=gather gran_scatter=scatter get_samples(m, sampler)
    while i != -1
        j = 0

        while j < i
            j += 1
            if is_important(v)
                @CUDA.cpu blocking=true types=(Nothing, Int64, CUDA.HostRef,) notify_important(v, sampler)
            end
            sync_warp()
            v += 1
        end

        (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) gran_gather=gather gran_scatter=scatter get_samples(m, sampler)
    end
    return
end

function kernel_gathered_non_blocking(m::Int64, sampler)
    (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) gran_gather=gather gran_scatter=scatter get_samples(m, sampler)
    while i != -1
        j = 0

        while j < i
            j += 1
            if is_important(v)
                @CUDA.cpu blocking=false types=(Nothing, Int64, CUDA.HostRef,) notify_important(v, sampler)
            end
            sync_warp()
            v += 1
        end

        (i, v) = @CUDA.cpu types=(Tuple{Int64, Int64}, Int64, CUDA.HostRef,) gran_gather=gather gran_scatter=scatter get_samples(m, sampler)
    end
    return
end

function kernel_cpu(m, sampler)
    (i, v) = get_samples(m, sampler)
    while i != -1
        j = 0

        while j < i
            j += 1
            if is_important(v)
                notify_important(v, sampler)
            end

            v += 1
        end

        (i, v) = get_samples(m, sampler)
    end
    return
end


function filter_bench(file=nothing; samples=4000000, step=100)
    if file === nothing
        file = "tmp/filterbench/test1.csv"
    end
    open(file, "w") do io
        perform_benchmarks([64, 128, 256], [1, 8, 16], [
            SimpleRunner("cpu", (_ns, _bs) -> ((sampler,) -> (kernel_cpu(step, sampler); println("Samples $(length(sampler.numbers)) at $(sampler.at)"))),
                quote (ns, bs) -> (Sampler($samples),) end),
            SimpleRunner("gpu_blocking", (ns, bs) -> ((sampler, manager, _) -> (@sync @cuda threads=ns blocks=bs manager=manager main_kernel_blocking(step, sampler))),
                quote (ns, bs) -> (s = Sampler($samples); (convert(CUDA.HostRef, s), CUDA.WarpAreaManager(8, 100), s,)) end),
            SimpleRunner("gpu_non_blocking", (ns, bs) -> ((sampler, manager, _) -> @sync @cuda threads=ns blocks=bs manager=manager  main_kernel_non_blocking(step, sampler)),
                quote (ns, bs) -> (s = Sampler($samples); (convert(CUDA.HostRef, s), CUDA.WarpAreaManager(8, 100), s,)) end),
            SimpleRunner("gpu_gather_blocking", (ns, bs) -> ((sampler, manager, _) -> @sync @cuda threads=ns blocks=bs manager=manager shmem=4*ns*sizeof(Int64) kernel_gathered_blocking(step, sampler)),
                quote (ns, bs) -> (s = Sampler($samples); (convert(CUDA.HostRef, s), CUDA.WarpAreaManager(8, 100), s,)) end),
            SimpleRunner("gpu_gather_non_blocking", (ns, bs) -> ((sampler, manager, s) -> (@sync @cuda threads=ns blocks=bs manager=manager shmem=4*ns*sizeof(Int64) kernel_gathered_non_blocking(step, sampler); println("Samples $(length(s.numbers)) at $(s.at)"))),
                quote (ns, bs) -> (s = Sampler($samples); (convert(CUDA.HostRef, s), CUDA.WarpAreaManager(8, 100), s,)) end),
        ]; io = io, samples=5, evals=10, measures=[:median, :std])
    end
end



function poller_bench(file=nothing; samples=4000000, step=100)
    if file === nothing
        file = "tmp/poller/test1.csv"
    end

    runners = []

    for manager in [CUDA.WarpAreaManager(16, 32 * 7, 32), CUDA.WarpAreaManager(32, 32 * 7, 32), CUDA.WarpAreaManager(64, 32 * 7, 32),
        CUDA.SimpleAreaManager(16, 32 * 7), CUDA.SimpleAreaManager(32, 32 * 7), CUDA.SimpleAreaManager(64, 32 * 7), CUDA.SimpleAreaManager(128, 32 * 7)]
        for poller in [CUDA.AlwaysPoller(0), CUDA.ConstantPoller(100), CUDA.ConstantPoller(500),
            CUDA.VarPoller([0, 50, 100, 500], CUDA.SaturationCounter), CUDA.VarPoller([0, 25, 50, 75, 100, 250, 500, 750, 1000], CUDA.SaturationCounter),
            CUDA.VarPoller([0, 50, 100, 500], CUDA.TwoLevelPredictor), CUDA.VarPoller([0, 25, 50, 75, 100, 250, 500, 750, 1000], CUDA.TwoLevelPredictor)]
            for policy in [CUDA.SimpleNotificationPolicy, CUDA.TreeNotificationPolicy{3}, CUDA.TreeNotificationPolicy{5}, CUDA.TreeNotificationPolicy{6}]
                runner = SimpleRunner(
                    "normal_$(shorten(manager))_$(shorten(poller))_$(shorten(policy))",
                    (ns, bs) -> ((sampler, _) -> @sync @cuda threads=ns blocks=bs manager=manager poller=poller policy_type=policy main_kernel_blocking(step, sampler)),
                    quote (ns, bs) -> (s = Sampler($samples); (convert(CUDA.HostRef, s), s,)) end
                )
                push!(runners, runner)

                runner = SimpleRunner(
                    "gathered_$(shorten(manager))_$(shorten(poller))_$(shorten(policy))",
                    (ns, bs) -> ((sampler, _) -> @sync @cuda threads=ns blocks=bs manager=manager poller=poller policy_type=policy shmem=4*ns*sizeof(Int64) kernel_gathered_blocking(step, sampler)),
                    quote (ns, bs) -> (s = Sampler($samples); (convert(CUDA.HostRef, s), s,)) end
                )
                push!(runners, runner)
            end
        end
    end

    open(file, "w") do io
        perform_benchmarks([128, 256], [64, 128, 256], runners; io = io, samples=3, evals=1, measures=[:median, :std])
    end
end
