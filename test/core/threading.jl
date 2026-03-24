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
    CUDACore.unsafe_free!(da)
    CUDACore.unsafe_free!(db)
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
      CUDACore.unsafe_free!(da)
      CUDACore.unsafe_free!(db)
    end
  end
end
