using GPUCompiler

# FIXME: these tests regularly triggers illegal memory accesses
#        after having moved to distributed test execution,
#        regardless of the memory pool or system.

@testset "threaded execution" begin
    function kernel(a, tid, id)
        a[1] = tid
        a[2] = id
        return
    end

    test_lock = ReentrantLock()
    Threads.@threads for id in 1:10
        da = CuArray{Int}(undef, 2)
        tid = Threads.threadid()
        @cuda kernel(da, tid, id)

        a = Array(da)
        lock(test_lock) do
            @test a == [tid, id]
        end
    end
end

@testset "concurrent compilation" begin
    # a kernel with a large unrolled body, so that ptxas takes a while
    @eval function slow_kernel(x)
        acc = x[1]
        $([:(acc = sin(acc * $(Float32(k))) + cos(acc)) for k in 1:200]...)
        x[1] = acc
        return
    end
    other_kernel(x) = (x[1] = 0; return)
    x = CuArray{Float32}(undef, 1)

    reports = Threads.Atomic{Int}(0)
    function hook(job)
        if job isa GPUCompiler.CompilerJob && job.source.def.name === :slow_kernel
            Threads.atomic_add!(reports, 1)
        end
    end
    kernels = GPUCompiler.with_compile_hook(hook) do
        tasks = [Threads.@spawn(@cuda launch=false slow_kernel(x)) for _ in 1:4]

        # ptxas runs without holding the lock, so we can take it while the slow kernel is
        # being assembled. holding on to it keeps that kernel from completing.
        lock(CUDACore.cufunction_lock)
        deadline = time() + 60
        while isempty(CUDACore.assembling_jobs) && time() < deadline
            unlock(CUDACore.cufunction_lock)
            sleep(0.01)
            lock(CUDACore.cufunction_lock)
        end
        try
            @test !isempty(CUDACore.assembling_jobs)

            # other kernels can still be compiled, assembling them under the lock
            @cuda launch=false other_kernel(x)
            @test !any(istaskdone, tasks)
        finally
            unlock(CUDACore.cufunction_lock)
        end

        fetch.(tasks)
    end
    # every task looked up the kernel, but it was only compiled once
    @test reports[] == length(kernels) + 1
    @test allequal(map(kernel -> kernel.fun, kernels))

    # tasks waiting on a failed compilation retry, and report their own error
    bad_kernel() = (Base.llvmcall("""call void asm sideeffect "invalid.instruction;", ""()
                                     ret void""", Cvoid, Tuple{}); return)
    tasks = map(1:4) do _
        Threads.@spawn try
            @cuda launch=false bad_kernel()
            nothing
        catch err
            err
        end
    end
    @test all(err -> occursin("Failed to compile PTX", sprint(showerror, err)), fetch.(tasks))
end

@testset "threaded arrays" begin
  test_lock = ReentrantLock()
  Threads.@threads for i in 1:Threads.nthreads()*100
    # allocates and uses unsafe_free to cover the allocator
    da = CuArray(rand(Float32, 64, 64))
    db = CuArray(rand(Float32, 64, 64))
    yield()
    dc = da .+ db
    yield()

    # @testset is not thread safe
    a = Array(da)
    b = Array(db)
    c = Array(dc)
    lock(test_lock) do
      @test c ≈ a .+ b
    end

    yield()
    CUDA.unsafe_free!(da)
    CUDA.unsafe_free!(db)
  end
end

@testset "threaded device usage" begin
  test_lock = ReentrantLock()
  Threads.@threads for i in 1:Threads.nthreads()*100
    dev = rand(1:length(devices()))
    device!(dev-1) do
      da = CuArray(rand(Float32, 64, 64))
      db = CuArray(rand(Float32, 64, 64))
      yield()
      dc = da .+ (db .* 2)
      yield()

      a = Array(da)
      b = Array(db)
      c = Array(dc)
      lock(test_lock) do
        @test c ≈ a .+ (b .* 2)
      end

      yield()
      CUDA.unsafe_free!(da)
      CUDA.unsafe_free!(db)
    end
  end
end
