
# Based on https://github.com/rocmarchive/Genesys_Syscall_test/blob/master/des.h

const sbox = UInt32[
	0x2, 0xe, 0xc, 0xb, 0x4, 0x2, 0x1, 0xc, 0x7, 0x4, 0xa, 0x7, 0xb, 0xd,
	0x6, 0x1, 0x8, 0x5, 0x5, 0x0, 0x3, 0xf, 0xf, 0xa, 0xb, 0x3, 0x0, 0x9,
	0xe, 0x8, 0x9, 0x6,
	0x4, 0xb, 0x2, 0x8, 0x1, 0xc, 0xb, 0x7, 0xa, 0x1, 0xb, 0xe, 0x7, 0x2,
	0x8, 0xd, 0xf, 0x6, 0x9, 0xf, 0xc, 0x0, 0x5, 0x9, 0x6, 0xa, 0x3, 0x4,
	0x0, 0x5, 0xe, 0x3
]

# This is not cryptographically significant
initial_perm(block::UInt64)::UInt64 = block

# This is not cryptographically significant
final_perm(block::UInt64)::UInt64 = block

lrotate28(a::UInt32)::UInt32 = ((a << 1) & 0xfffffff) | ((a >> 27) & 0x1);

function subkey(key::UInt64, i)::UInt64
    top = UInt32((key >> 32) & 0xfffffff)
    bottom = UInt32(key & 0xfffffff)

    for _ in 1:i
        top = lrotate28(top)
        bottom = lrotate28(bottom)
    end

    sk = (UInt64(top) << 24) | UInt64(bottom)

    return sk
end

function faisal(halfblock::UInt32, subkey::UInt64, sbox)::UInt32
    expanded = UInt64(0)
    for i in 0:7
        expanded |= (UInt64(halfblock) << (i * 2 + 1)) & (UInt64(0x3f) << (i * 6))
    end

    halfblock = UInt32(0)

    expanded = xor(expanded, subkey)

    for i in 0:7
        halfblock |= UInt32(((expanded >> (i * 6)) & 0x3f) << (i * 4) & 0xfffffff)
        halfblock |= sbox[((expanded >> (i * 6)) & 0x3f) + 0x1] << (i * 4)
    end

	return halfblock;
end

function run_des(block::UInt64, key::UInt64, sbox)
    block = initial_perm(block)

    a = UInt32((block >> 32) & 0xfffffff)
    b = UInt32(block & 0xfffffff)
    for i in 0:15
        tmp = xor(a, faisal(b, subkey(key, i), sbox))
        a = b
        b = tmp
    end

    block = (UInt64(a) << 32) | UInt64(b)
    return final_perm(block)
end

using CUDA
using BenchmarkTools
using LLVM
using LLVM.Interop
using Core: LLVMPtr
using Random

mutable struct EncryptionManager
    at::Int64
    total::Int64
end

function get_new_value(manager, arr, count)
    manager === nothing && return 0
    manager.at == manager.total && return 0

    if manager.at + count > manager.total
        count = manager.total - manager.at
        manager.at = manager.total
    else
        manager.at += count
    end

    ta = reinterpret(Ptr{UInt64}, arr)

    for i in 1:count
        unsafe_store!(ta, rand(UInt64), i)
    end

    return count
end



function get_new_value_gpu_normal(manager, p, index)
    old_v = unsafe_load(p)
    returned = @CUDA.cpu types=(Int64, HostRef, CuPtr{UInt64}, Int64) get_new_value(manager, p, 1)
    returned > 0 || return false

    new_v = unsafe_load(p)

    tc = 0
    while old_v == new_v && tc < 5
        nanosleep(UInt32(32))
        new_v = unsafe_load(p)
        tc += 1
    end

    tc == 5 && @cuprintln("$index: Value not updated (type 1)")

    true
end

des_gather((m1, p1, v1), (_, p2, v2)) = (v1, (m1, min(p1, p2), v1 + v2))

des_scatter(state, t) = (min(t, state), t-state)
function get_new_value_gpu_gather(manager, p, index)
    old_v = unsafe_load(p)

    returned = @CUDA.cpu types=(Int64, HostRef, CuPtr{UInt64}, Int64) gran_gather=des_gather gran_scatter=des_scatter get_new_value(manager, p, 1)
    returned > 0 || return false

    new_v = unsafe_load(p)

    tc = 0
    while old_v == new_v && tc < 5
        nanosleep(UInt32(32))
        new_v = unsafe_load(p)
        tc += 1
    end

    tc == 5 && @cuprintln("$index: Value not updated (type 1)")

    true
end

function return_new_value(arr, count)
    ta = reinterpret(Ptr{UInt64}, arr)
    for i in 1:count
        v = unsafe_load(ta, i)

        # Do something with the encrypted value
    end
end

function return_new_value_gpu_blocking(p)
    @CUDA.cpu types=(Nothing, CuPtr{UInt64}, Int64) return_new_value(p, 1)
end

function return_new_value_gpu_non_blocking(p)
    @CUDA.cpu blocking=false types=(Nothing, CuPtr{UInt64}, Int64) return_new_value(p, 1)
end




@inline function get_thread_id()
    return (blockIdx().x - 1) * blockDim().x + threadIdx().x
end


function des_kernel(manager, in, out, iterations, in_f, out_f, key, sbox)
    index = get_thread_id()
    p_in = pointer(in, index)
    p_out = pointer(out, index)

    while in_f(manager, p_in, index)
        sync_warp()

        v = in[index]

        for _ in 1:iterations
            v = run_des(v, key, sbox)
        end

        out[index] = v

        out_f(p_out)
    end

    return nothing
end

function des_kernel_2(manager, in, out, iterations, in_f, out_f, key, sbox)
    index = get_thread_id()
    p_in = pointer(in, index)
    p_out = pointer(out, index)

    while in_f(manager, p_in, index)
        sync_warp()

        v = in[index]

        for _ in 1:iterations
            v = run_des(v, key, sbox)
        end

        out[index] = v

        if v % 100 == 0
            out_f(p_out)
        end
    end

    return nothing
end

arr_in = nothing
arr_out = nothing


function get_bench_requirements(threads, blocks; total=32 * 1024)
    global arr_in
    global arr_out

    membuffer_in = CUDA.Mem.alloc(Mem.Host, threads*blocks * sizeof(UInt64), Mem.HOSTALLOC_DEVICEMAP | Mem.HOSTALLOC_WRITECOMBINED)
    arr_in = CUDA.unsafe_wrap(CuArray{UInt64}, reinterpret(CuPtr{UInt64}, pointer(membuffer_in)), threads*blocks)

    membuffer_out = CUDA.Mem.alloc(Mem.Host, threads*blocks * sizeof(UInt64), Mem.HOSTALLOC_DEVICEMAP | Mem.HOSTALLOC_WRITECOMBINED)
    arr_out = CUDA.unsafe_wrap(CuArray{UInt64}, reinterpret(CuPtr{UInt64}, pointer(membuffer_out)), threads*blocks)

    enc_manager = EncryptionManager(0, total)
    enc_manager_ref = convert(CUDA.HostRef, enc_manager)

    cu_sbox = CuArray(sbox)
    key = rand(UInt64)

    (arr_in, arr_out, enc_manager, enc_manager_ref, cu_sbox, key)
end


function des_bench_first(file=nothing, poller_count = nothing)
    if file === nothing
        file = "tmp/des/test1.csv"
    end

    if poller_count === nothing
        poller_count = 1
    end

    if isa(poller_count, String)
        poller_count = parse(Int64, poller_count)
    end

    managers = [(CUDA.SimpleAreaManager(32, 128), "simple(32)"), (CUDA.SimpleAreaManager(64, 128), "simple(64)"), (CUDA.SimpleAreaManager(128, 128), "simple(128)"), (CUDA.SimpleAreaManager(256, 128), "simple(256)"),
                (CUDA.SimpleAreaManager(512, 128), "simple(512)"),
                (CUDA.WarpAreaManager(1, 128), "warp(1)"), (CUDA.WarpAreaManager(4, 128), "warp(4)"), (CUDA.WarpAreaManager(4, 128), "warp(4)"), (CUDA.WarpAreaManager(8, 128), "warp(8)"), (CUDA.WarpAreaManager(16, 128), "warp(16)"), (CUDA.WarpAreaManager(32, 128), "warp(32)")]
    policies = [(CUDA.SimpleNotificationPolicy, "simple"), (CUDA.TreeNotificationPolicy{3}, "tree(3)"), (CUDA.TreeNotificationPolicy{6}, "tree(6)")]

    threads = 128
    blocks = 32
    samples = 10

    (arr_in, arr_out, enc_manager, enc_manager_ref, cu_sbox, key) = get_bench_requirements(threads, blocks)

    function run_kernel(manager, policy, iterations)
        @sync @cuda manager=manager policy_type=policy poller_count=poller_count threads=threads blocks=blocks des_kernel(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_normal, return_new_value_gpu_blocking, key, cu_sbox)
    end

    open(file, "w") do io
        print(io, "x")
        for (_, manager) in managers
            for (_, policy) in policies
                print(io, ",$(trial_header("$(manager)_$(policy)", [:median, :std]))")
            end
        end
        println(io, "")
        flush(io)

        for iteration in 25:10:512
            print(io, "$iteration")
            for (manager, _) in managers
                for (policy, _) in policies
                    trial = @benchmark ($run_kernel($manager, $policy, $iteration)) setup=($enc_manager.at = 0) samples=samples evals=1
                    times = [x / iteration for x in trial.times]
                    print(io, ",$(times_to_string(times, [:median, :std]))")
                    flush(io)
                end
            end
            println(io, "")
            flush(io)
        end
    end
end

function des_bench_2(file=nothing, poller_count = nothing)
    if file === nothing
        file = "tmp/des/test1_2.csv"
    end

    if poller_count === nothing
        poller_count = 1
    end

    if isa(poller_count, String)
        poller_count = parse(Int64, poller_count)
    end

    managers = [(CUDA.SimpleAreaManager(32, 128), "simple(32)"), (CUDA.SimpleAreaManager(64, 128), "simple(64)"), (CUDA.SimpleAreaManager(128, 128), "simple(128)"), (CUDA.SimpleAreaManager(256, 128), "simple(256)"),
                (CUDA.SimpleAreaManager(512, 128), "simple(512)"),
                (CUDA.WarpAreaManager(1, 128), "warp(1)"), (CUDA.WarpAreaManager(4, 128), "warp(4)"), (CUDA.WarpAreaManager(8, 128), "warp(8)"), (CUDA.WarpAreaManager(16, 128), "warp(16)"), (CUDA.WarpAreaManager(32, 128), "warp(32)")]
    policies = [(CUDA.SimpleNotificationPolicy, "simple"), (CUDA.TreeNotificationPolicy{3}, "tree(3)"), (CUDA.TreeNotificationPolicy{6}, "tree(6)")]

    threads = 128
    blocks = 32
    samples = 10

    (arr_in, arr_out, enc_manager, enc_manager_ref, cu_sbox, key) = get_bench_requirements(threads, blocks)


    function run_kernel(manager, policy, iterations)
        @sync @cuda manager=manager policy_type=policy poller_count=poller_count threads=threads blocks=blocks des_kernel_2(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_normal, return_new_value_gpu_blocking, key, cu_sbox)
    end

    open(file, "w") do io
        print(io, "x")
        for (_, manager) in managers
            for (_, policy) in policies
                print(io, ",$(trial_header("$(manager)_$(policy)", [:median, :std]))")
            end
        end
        println(io, "")
        flush(io)

        for iteration in 25:15:256
            print(io, "$iteration")
            for (manager, _) in managers
                for (policy, _) in policies
                    trial = @benchmark ($run_kernel($manager, $policy, $iteration)) setup=($enc_manager.at = 0) samples=samples evals=1
                    times = [x / iteration for x in trial.times]
                    print(io, ",$(times_to_string(times, [:median, :std]))")
                    flush(io)
                end
            end
            println(io, "")
            flush(io)
        end
    end
end


function des_bench_poller(file=nothing)
    if file === nothing
        file = "tmp/des/poller_test.csv"
    end

    managers = [(CUDA.SimpleAreaManager(256, 128), "simple256"), (CUDA.WarpAreaManager(8, 128), "warp8")]

    policies = [(CUDA.SimpleNotificationPolicy, "simple"), (CUDA.TreeNotificationPolicy{3}, "tree(3)"), (CUDA.TreeNotificationPolicy{6}, "tree(6)")]

    pollers = [(CUDA.AlwaysPoller(0), "const0"), (CUDA.ConstantPoller(100), "const100"), (CUDA.ConstantPoller(500), "const500"),
        (CUDA.VarPoller([0, 50, 100, 500], CUDA.SaturationCounter), "varsat4"), (CUDA.VarPoller([0, 25, 50, 75, 100, 250, 500, 750, 1000], CUDA.SaturationCounter), "varsat9"),
        (CUDA.VarPoller([0, 50, 100, 500], CUDA.TwoLevelPredictor), "vartwo4"), (CUDA.VarPoller([0, 25, 50, 75, 100, 250, 500, 750, 1000], CUDA.TwoLevelPredictor), "vartwo9")]

    threads = 128
    blocks = 32
    samples = 10

    (arr_in, arr_out, enc_manager, enc_manager_ref, cu_sbox, key) = get_bench_requirements(threads, blocks)


    function run_kernel(manager, policy, poller, iterations)
        @sync @cuda manager=manager policy_type=policy poller=poller threads=threads blocks=blocks des_kernel(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_normal, return_new_value_gpu_blocking, key, cu_sbox)
    end

    open(file, "w") do io
        print(io, "x")
        for (_, manager) in managers
            for (_, policy) in policies
                for (_,poller) in pollers
                    print(io, ",$(trial_header("$(manager)_$(policy)_$(poller)_time", [:median, :std]))")
                    print(io, ",$(trial_header("$(manager)_$(policy)_$(poller)_cpu", [:median, :std]))")
                end
            end
        end
        println(io, "")
        flush(io)

        for iteration in 25:15:256
            print(io, "$iteration")
            for (manager, _) in managers
                for (policy, _) in policies
                    for (poller,_) in pollers
                        trial = @benchmark ($run_kernel($manager, $policy, $poller, $iteration)) setup=($enc_manager.at = 0) samples=samples evals=1
                        times = [x / iteration for x in trial.times]
                        print(io, ",$(times_to_string(times, [:median, :std]))")

                        cpu_times = []
                        for i in 1:samples
                            enc_manager.at = 0; GC.gc()

                            time = run_kernel(manager, policy, poller, iteration)
                            push!(cpu_times, time / iteration)
                            # push!(hit_times, sum(hits))
                            # push!(miss_times, sum(misses))
                        end
                        print(io, ",$(times_to_string_unchanged(cpu_times, [:median, :std]))")

                        flush(io)
                    end
                end
            end
            println(io, "")
            flush(io)
        end

    end

end


function des_bench_poller_2(file=nothing)
    if file === nothing
        file = "tmp/des/poller_test_2.csv"
    end

    managers = [(CUDA.SimpleAreaManager(256, 128), "simple256"), (CUDA.WarpAreaManager(8, 128), "warp8")]

    policies = [(CUDA.SimpleNotificationPolicy, "simple"), (CUDA.TreeNotificationPolicy{3}, "tree(3)"), (CUDA.TreeNotificationPolicy{6}, "tree(6)")]

    pollers = [(CUDA.AlwaysPoller(0), "const0"), (CUDA.ConstantPoller(100), "const100"), (CUDA.ConstantPoller(500), "const500"),
        (CUDA.VarPoller([0, 50, 100, 500], CUDA.SaturationCounter), "varsat4"), (CUDA.VarPoller([0, 25, 50, 75, 100, 250, 500, 750, 1000], CUDA.SaturationCounter), "varsat9"),
        (CUDA.VarPoller([0, 50, 100, 500], CUDA.TwoLevelPredictor), "vartwo4"), (CUDA.VarPoller([0, 25, 50, 75, 100, 250, 500, 750, 1000], CUDA.TwoLevelPredictor), "vartwo9")]

    threads = 128
    blocks = 32
    samples = 10

    (arr_in, arr_out, enc_manager, enc_manager_ref, cu_sbox, key) = get_bench_requirements(threads, blocks)


    function run_kernel(manager, policy, poller, iterations)
        @sync @cuda manager=manager policy_type=policy poller=poller threads=threads blocks=blocks des_kernel_2(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_normal, return_new_value_gpu_blocking, key, cu_sbox)
    end

    open(file, "w") do io
        print(io, "x")
        for (_, manager) in managers
            for (_, policy) in policies
                for (_,poller) in pollers
                    print(io, ",$(trial_header("$(manager)_$(policy)_$(poller)_time", [:median, :std]))")
                    print(io, ",$(trial_header("$(manager)_$(policy)_$(poller)_cpu", [:median, :std]))")
                end
            end
        end
        println(io, "")
        flush(io)

        for iteration in 25:15:256
            print(io, "$iteration")
            for (manager, _) in managers
                for (policy, _) in policies
                    for (poller,_) in pollers
                        trial = @benchmark ($run_kernel($manager, $policy, $poller, $iteration)) setup=($enc_manager.at = 0) samples=samples evals=1
                        times = [x / iteration for x in trial.times]
                        print(io, ",$(times_to_string(times, [:median, :std]))")

                        cpu_times = []
                        for i in 1:samples
                            enc_manager.at = 0; GC.gc()

                            time = run_kernel(manager, policy, poller, iteration)
                            push!(cpu_times, time / iteration)
                        end
                        print(io, ",$(times_to_string_unchanged(cpu_times, [:median, :std]))")

                        flush(io)
                    end
                end
            end
            println(io, "")
            flush(io)
        end
    end
end


function des_bench_total(file=nothing)
    if file === nothing
        file = "tmp/des3/test1.csv"
    end

    managers = [(CUDA.SimpleAreaManager(32, 128), "simple(32)"), (CUDA.SimpleAreaManager(64, 128), "simple(64)"), (CUDA.SimpleAreaManager(128, 128), "simple(128)"), (CUDA.SimpleAreaManager(256, 128), "simple(256)"),
                (CUDA.SimpleAreaManager(512, 128), "simple(512)"),
                (CUDA.WarpAreaManager(1, 128), "warp(1)"), (CUDA.WarpAreaManager(4, 128), "warp(4)"), (CUDA.WarpAreaManager(4, 128), "warp(4)"), (CUDA.WarpAreaManager(8, 128), "warp(8)"), (CUDA.WarpAreaManager(16, 128), "warp(16)"), (CUDA.WarpAreaManager(32, 128), "warp(32)")]


    threads = 128
    blocks = 32
    samples = 10

    (arr_in, arr_out, enc_manager, enc_manager_ref, cu_sbox, key) = get_bench_requirements(threads, blocks)

    function run_kernel(manager, iterations)
        @sync @cuda manager=manager threads=threads blocks=blocks des_kernel(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_normal, return_new_value_gpu_blocking, key, cu_sbox)
    end

    function run_kernel_non_blocking(manager, iterations)
        @sync @cuda manager=manager threads=threads blocks=blocks des_kernel(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_normal, return_new_value_gpu_non_blocking, key, cu_sbox)
    end

    function run_kernel_gather(manager, iterations)
        @sync @cuda manager=manager threads=threads blocks=blocks shmem=128*4*sizeof(Int64) des_kernel(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_gather, return_new_value_gpu_blocking, key, cu_sbox)
    end

    function run_kernel_non_blocking_and_gather(manager, iterations)
        @sync @cuda manager=manager threads=threads blocks=blocks shmem=128*4*sizeof(Int64) des_kernel(enc_manager_ref, arr_in, arr_out, iterations, get_new_value_gpu_gather, return_new_value_gpu_non_blocking, key, cu_sbox)
    end

    kernels = [(run_kernel, "normal"), (run_kernel_non_blocking, "non_blocking"), (run_kernel_gather, "gather"),  (run_kernel_non_blocking_and_gather, "non_blocking_and_gather")]


    open(file, "w") do io
        print(io, "x")
        for (_, manager) in managers
            for (_, f) in kernels
                print(io, ",$(trial_header("$(manager)_$(f)", [:median, :std]))")
            end
        end
        println(io, "")
        flush(io)

        for iteration in 25:10:512
            print(io, "$iteration")
            for (manager, _) in managers
                for (f, _) in kernels
                    trial = @benchmark ($f($manager, $iteration)) setup=($enc_manager.at = 0) samples=samples evals=1
                    times = [x / iteration for x in trial.times]
                    print(io, ",$(times_to_string(times, [:median, :std]))")
                    flush(io)
                end
            end
            println(io, "")
            flush(io)
        end
    end
end
