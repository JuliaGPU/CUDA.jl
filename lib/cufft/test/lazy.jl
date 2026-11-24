using LazyArrays
using Base.CoreLogging: Debug, with_logger

# broadcasts around `@~ p * x` are fused into the transform where cuFFT supports callbacks,
# and executed after the transform otherwise, with the same results
# (not with forced bounds checking, as broadcasts that may throw aren't fused)
supported = cuFFT.callback_issue(plan_fft(CuArray{ComplexF32}(undef, 4))) === nothing &&
            Base.JLOptions().check_bounds != 1

# evaluate `f()`, returning its result and the reasons why broadcasts weren't fused
function fusion(f)
    logger = Test.TestLogger(min_level=Debug)
    result = with_logger(f, logger)
    reasons = String[log.kwargs[:reason] for log in logger.logs
                     if startswith(string(log.message), "cuFFT: not fusing")]
    return result, reasons
end

# test that `ex` fuses its broadcast (unless that is disabled, or not supported at all)
macro fused(ex)
    quote
        result, reasons = fusion(() -> $(esc(ex)))
        if supported && cuFFT.fuse_broadcasts[]
            @test isempty(reasons)
        else
            @test length(reasons) == 1
        end
        result
    end
end

# test that `ex` doesn't fuse its broadcast, for a reason containing `why`
macro unfused(why, ex)
    quote
        result, reasons = fusion(() -> $(esc(ex)))
        @test length(reasons) == 1
        if supported && cuFFT.fuse_broadcasts[]
            @test occursin($(esc(why)), only(reasons))
        end
        result
    end
end

function with_fusion(f, enabled::Bool)
    previous = cuFFT.fuse_broadcasts[]
    cuFFT.fuse_broadcasts[] = enabled
    try
        f()
    finally
        cuFFT.fuse_broadcasts[] = previous
    end
end

lazy_helper(x) = 2x
@noinline pick_type(x) = x > 1 ? Float32 : Float64
may_throw(x) = x > 1f30 ? throw(DomainError(x)) : x

@testset "fusion $(fuse ? "enabled" : "disabled")" for fuse in (true, false)
with_fusion(fuse) do

@testset "$T" for T in (Float32, Float64)
    x = rand(Complex{T}, 64, 8)
    w = rand(T, 64)
    d_x, d_w = CuArray(x), CuArray(w)
    p = plan_fft(d_x, 1)
    y = fft(w .* x, 1)

    @testset "load" begin
        @test Array(@fused copy(@~ p * (d_w .* d_x))) ≈ y
        @test Array(@fused LazyArrays.materialize(@~ p * (d_w .* d_x))) ≈ y

        # arrays of the input's size, scalars, tuples and nested calls
        full = rand(T, 64, 8)
        d_full = CuArray(full)
        @test Array(@fused copy(@~ p * (d_full .* d_x .+ 1))) ≈ fft(full .* x .+ 1, 1)
        row = rand(T, 1, 8)
        d_row = CuArray(row)
        @test Array(@fused copy(@~ p * (d_w .* d_x .* d_row))) ≈ fft(w .* x .* row, 1)
        @test Array(@fused copy(@~ p * muladd.(d_x, 2, d_w))) ≈ fft(muladd.(x, 2, w), 1)

        # inputs of another element type are converted while loading
        s = rand(Int16, 64, 8)
        d_s = CuArray(s)
        @test Array(@fused copy(@~ p * d_s)) ≈ fft(s, 1)
        @test Array(@fused copy(@~ p * (d_w .* d_s))) ≈ fft(w .* s, 1)
        c = rand(Complex{Int8}, 64, 8)
        d_c = CuArray(c)
        @test Array(@fused copy(@~ p * d_c)) ≈ fft(c, 1)
        @test Array(@fused copy(@~ p * Complex{T}.(d_s, 1))) ≈ fft(Complex{T}.(s, 1), 1)
    end

    @testset "store" begin
        n = T(64)
        @test Array(@fused (@~ p * d_x) ./ n) ≈ fft(x, 1) ./ n
        @test Array(@fused (@~ p * d_x) .* d_w) ≈ fft(x, 1) .* w
        @test Array(@fused muladd.(@~(p * d_x), d_x, 1)) ≈ muladd.(fft(x, 1), x, 1)

        # results of another type
        power = @fused abs2.(@~ p * d_x)
        @test power isa CuArray{T}
        @test Array(power) ≈ abs2.(fft(x, 1))
        wide = @fused ComplexF64.(@~ p * d_x)
        @test wide isa CuArray{ComplexF64}
        @test Array(wide) ≈ fft(x, 1)
    end

    @testset "load and store" begin
        n = T(64)
        @test Array(@fused abs2.(@~ p * (d_w .* d_x)) ./ n) ≈ abs2.(y) ./ n
    end

    @testset "destinations" begin
        d_y = similar(d_x)
        @fused d_y .= (@~ p * (d_w .* d_x)) ./ 2
        @test Array(d_y) ≈ y ./ 2
        @fused d_y .= @~ p * (d_w .* d_x)
        @test Array(d_y) ≈ y

        # the destination can be part of the broadcast
        acc = ones(T, 64, 8)
        d_acc = CuArray(acc)
        @fused d_acc .+= abs2.(@~ p * d_x)
        @fused d_acc .+= abs2.(@~ p * d_x)
        @test Array(d_acc) ≈ acc .+ 2 .* abs2.(fft(x, 1))
        @fused d_y .= d_y .- (@~ p * (d_w .* d_x))
        @test maximum(abs, Array(d_y)) < 1e-3

        # and have another element type
        d_z = CuArray{ComplexF64}(undef, 64, 8)
        @fused d_z .= @~ p * (d_w .* d_x)
        @test Array(d_z) ≈ y
    end

    @testset "normalization" begin
        q = plan_ifft(d_x, 1)
        @test Array(@fused copy(@~ q * d_x)) ≈ ifft(x, 1)
        @test Array(@fused real.(@~ q * (d_w .* d_x))) ≈ real.(ifft(w .* x, 1))
        q = inv(p)
        @test Array(@fused abs2.(@~ q * d_x)) ≈ abs2.(ifft(x, 1))
    end

    @testset "dimensions" begin
        for region in (2, (1, 2))
            q = plan_fft(d_x, region)
            @test Array(@fused abs2.(@~ q * (d_w .* d_x))) ≈ abs2.(fft(w .* x, region))
        end
        v = rand(Complex{T}, 6, 10, 12)
        d_v = CuArray(v)
        for region in (1, (1, 2), (2, 3), (1, 2, 3))
            q = plan_fft(d_v, region)
            @test Array(@fused abs2.(@~ q * (2 .* d_v))) ≈ abs2.(fft(2 .* v, region))
            q = plan_bfft(d_v, region)
            @test Array(@fused (@~ q * (2 .* d_v)) ./ 3) ≈ bfft(2 .* v, region) ./ 3
        end
    end

    @testset "real" begin
        r = rand(T, 64, 8)
        d_r = CuArray(r)
        q = plan_rfft(d_r, 1)
        @test Array(@fused abs2.(@~ q * (d_w .* d_r))) ≈ abs2.(rfft(w .* r, 1))
        @test Array(@fused abs2.(@~ q * d_r)) ≈ abs2.(rfft(r, 1))
        r2 = plan_rfft(d_r)
        @test Array(@fused copy(@~ r2 * (d_w .* d_r))) ≈ rfft(w .* r)

        # complex-to-real transforms only fuse the broadcast around them, and leave their
        # input alone
        c = rfft(r, 1)
        d_c = CuArray(c)
        b = plan_brfft(d_c, 64, 1)
        @test Array(@fused (@~ b * d_c) ./ 64) ≈ brfft(c, 64, 1) ./ 64
        @test Array(d_c) == c
        @test Array(@fused d_w .* (@~ b * (2 .* d_c))) ≈ w .* brfft(2 .* c, 64, 1)
        i = plan_irfft(d_c, 64, 1)
        @test Array(@fused copy(@~ i * d_c)) ≈ r
        @test Array(@fused d_w .* (@~ i * (2 .* d_c))) ≈ w .* 2 .* r
    end

    @testset "scratch space" begin
        # cuFFT uses its output array as scratch space for some sizes, which must not
        # reach a destination that is smaller than the transform's output, or that the
        # broadcast reads
        for (sz, region) in (((30, 20), (1, 2)), ((2^9 * 113,), 1), ((2 * 3 * 5 * 7 * 11 * 13,), 1))
            v = rand(Complex{T}, sz...)
            d_v = CuArray(v)
            q = plan_fft(d_v, region)
            len = length(v)
            buffer = CUDACore.fill(T(-1), 3len)
            dest = reshape(view(buffer, len+1:2len), sz)
            dest .= 1
            @fused dest .+= abs2.(@~ q * (2 .* d_v))
            @test Array(dest) ≈ 1 .+ abs2.(fft(2 .* v, region))
            @test all(==(-1), Array(buffer)[1:len])
            @test all(==(-1), Array(buffer)[2len+1:end])
        end
    end

    @testset "sizes" begin
        for n in (60, 113, 126, 1000)
            v = rand(Complex{T}, n, 3)
            d_v = CuArray(v)
            q = plan_fft(d_v, 1)
            @test Array(@fused abs2.(@~ q * (2 .* d_v))) ≈ abs2.(fft(2 .* v, 1))
        end
    end
end

@testset "state" begin
    # broadcasts of the same type share compiled code, but not their arguments
    x = rand(ComplexF32, 256)
    d_x = CuArray(x)
    p = plan_fft(d_x)
    windowed(d_w) = copy(@~ p * (d_w .* d_x))
    w1, w2 = rand(Float32, 256), rand(Float32, 256)
    @test Array(@fused windowed(CuArray(w1))) ≈ fft(w1 .* x)
    @test Array(@fused windowed(CuArray(w2))) ≈ fft(w2 .* x)
    scaled(s) = (@~ p * d_x) .* s
    @test Array(@fused scaled(2f0)) ≈ 2 .* fft(x)
    @test Array(@fused scaled(3f0)) ≈ 3 .* fft(x)

    # also on other streams
    results = map(1:4) do s
        Threads.@spawn Array(scaled(Float32(s)))
    end
    for (s, t) in enumerate(results)
        @test fetch(t) ≈ s .* fft(x)
    end
end

@testset "concurrent executions" begin
    # one plan, used by several tasks on their own streams
    w = rand(Float32, 1024, 64)
    d_w = CuArray(w)
    p = plan_fft(CuArray{ComplexF32}(undef, 1024, 64), 1)
    xs = [rand(ComplexF32, 1024, 64) for _ in 1:4]
    tasks = map(xs) do x
        Threads.@spawn begin
            d_x = CuArray(x)
            d_y = similar(d_x)
            for _ in 1:3
                d_y .= 2 .* (@~ p * (d_w .* d_x))
            end
            Array(d_y)
        end
    end
    for (x, t) in zip(xs, tasks)
        @test fetch(t) ≈ 2 .* fft(x .* w, 1)
    end
end

@testset "garbage collection" begin
    # arrays that only the broadcast references
    x = rand(ComplexF32, 4096, 64)
    w = rand(Float32, 4096, 64)
    d_x = CuArray(x)
    d_y = similar(d_x)
    p = plan_fft(d_x, 1)
    let d_w = CuArray(w)
        d_y .= @~ p * (d_w .* d_x)
    end
    GC.gc(true)
    GC.gc(true)
    @test Array(d_y) ≈ fft(x .* w, 1)
end

@testset "redefinition" begin
    x = rand(ComplexF32, 64)
    d_x = CuArray(x)
    p = plan_fft(d_x)
    f() = lazy_helper.(@~ p * d_x)
    @eval lazy_helper(x) = 2x
    @test Array(Base.invokelatest(f)) ≈ 2 .* fft(x)
    @eval lazy_helper(x) = 3x
    @test Array(Base.invokelatest(f)) ≈ 3 .* fft(x)
end

@testset "fallbacks" begin
    x = rand(ComplexF32, 64, 8)
    w = rand(Float32, 64)
    d_x, d_w = CuArray(x), CuArray(w)
    p = plan_fft(d_x, 1)
    y = fft(x, 1)

    # a plain transform
    @test Array(@unfused "nothing to fuse" copy(@~ p * d_x)) ≈ y
    d_y = similar(d_x)
    @unfused "nothing to fuse" d_y .= @~ p * d_x
    @test Array(d_y) ≈ y

    # more than one transform
    @test Array(@unfused "more than one" (@~ p * d_x) .+ (@~ p * d_x)) ≈ 2 .* y

    # in-place plans
    q = plan_fft!(similar(d_x), 1)
    @test Array(@unfused "in-place" abs2.(@~ q * (d_w .* d_x))) ≈ abs2.(fft(w .* x, 1))

    # half precision
    h = rand(ComplexF16, 64, 8)
    d_h = CuArray(h)
    q = plan_fft(d_h, 1)
    @test Array(@unfused "precision" abs2.(@~ q * (2 .* d_h))) ≈ abs2.(fft(2 .* h, 1)) rtol=1e-2

    # layouts that cuFFT cannot batch
    v = rand(ComplexF32, 4, 5, 6)
    d_v = CuArray(v)
    q = plan_fft(d_v, 2)
    @test Array(@unfused "permuting" abs2.(@~ q * (2 .* d_v))) ≈ abs2.(fft(2 .* v, 2))

    # sizes that cuFFT doesn't support callbacks for
    z = rand(ComplexF32, 127, 4)
    d_z = CuArray(z)
    q = plan_fft(d_z, 1)
    @test Array(@unfused "prime factor" abs2.(@~ q * (2 .* d_z))) ≈ abs2.(fft(2 .* z, 1))

    # transforms that cuFFT doesn't call callbacks for (large batches of small 2D complex
    # transforms, as of cuFFT 12.5)
    b = rand(ComplexF32, 16, 16, 4096)
    d_b = CuArray(b)
    q = plan_fft(d_b, (1, 2))
    result, _ = fusion(() -> abs2.(@~ q * (2 .* d_b)))
    @test Array(result) ≈ abs2.(fft(2 .* b, (1, 2)))

    # destinations that aren't dense
    d_big = CUDACore.zeros(ComplexF32, 128, 8)
    @unfused "dense" view(d_big, 1:2:128, :) .= 2 .* (@~ p * d_x)
    @test Array(d_big)[1:2:128, :] ≈ 2 .* y

    # broadcasts that change the shape
    k = rand(Float32, 1, 1, 3)
    d_k = CuArray(k)
    @test Array(@unfused "shape" (@~ p * d_x) .* d_k) ≈ y .* k

    # destinations that alias an array the broadcast reads
    d_y = copy(d_x)
    @unfused "aliases" d_y .= @~ p * (d_w .* d_y)
    @test Array(d_y) ≈ fft(w .* x, 1)
    d_y = copy(d_x)
    @unfused "aliases" d_y .= (@~ p * d_x) .* view(d_y, 64:-1:1, :)
    @test Array(d_y) ≈ y .* x[64:-1:1, :]

    # broadcasts that may throw
    @test Array(@unfused "compiled" may_throw.(abs2.(@~ p * d_x))) ≈ abs2.(y)
    @test Array(@unfused "compiled" copy(@~ p * (may_throw.(d_w) .* d_x))) ≈ fft(w .* x, 1)
    d_r = CUDACore.zeros(Float32, 64, 8)
    @test_throws Exception (d_r .= @~ p * d_x; synchronize())

    # or refer to objects that CUDA.jl patches into kernels when loading them (which
    # depends on the Julia version)
    patched(v) = v * (pick_type(real(v)) === Float32 ? 2f0 : 1f0)
    result, _ = fusion(() -> patched.(@~ p * d_x))
    @test Array(result) ≈ patched.(y)

    # graph capture
    d_y = similar(d_x)
    graph = CUDACore.capture() do
        @unfused "captured" d_y .= 2 .* (@~ p * (d_w .* d_x))
    end
    exec = CUDACore.instantiate(graph)
    exec()
    @test Array(d_y) ≈ 2 .* fft(w .* x, 1)
end

end
end
