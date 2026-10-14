# A strided destination with sentinels outside the selected rows.
function strided_destination(T, dims)
    storage = CuArray(ones(T, (2dims[1], dims[2:end]...)))
    indices = (2:2:size(storage, 1), ntuple(_ -> Colon(), length(dims)-1)...)
    outside = trues(size(storage))
    view(outside, indices...) .= false
    view(storage, indices...), outside
end
view_matches(x, expected) = isapprox(Array(x), expected;
    rtol=rtol(eltype(x)), atol=atol(eltype(x)))

@testset "complex FFT views" begin
    CUDACore.allowscalar(false)
    layouts = (("row", (2, :, 1), nothing),
               ("reversed offset", (8:-2:2, 2:5, 1), nothing),
               ("contiguous", (:, :, 1), nothing),
               ("external batches", (1:4, :, :), (2,)))
    @testset "$T, $name" for T in (ComplexF16, ComplexF32, ComplexF64),
                            (name, indices, region) in layouts
        host = rand(T, 8, 8, 4)
        device = CuArray(host)
        x = view(device, indices...)
        input = Array(view(host, indices...))
        args = isnothing(region) ? () : (region,)
        dims = isnothing(region) ? ntuple(identity, ndims(x)) : region
        outside = trues(size(host))
        view(outside, indices...) .= false
        for planner in (plan_fft, plan_bfft, plan_fft!, plan_bfft!)
            @test which(planner, (typeof(x),)).module === cuFFT
        end
        for (f, pf, f!, pf!) in ((fft, plan_fft, fft!, plan_fft!),
                                 (bfft, plan_bfft, bfft!, plan_bfft!),
                                 (ifft, plan_ifft, ifft!, plan_ifft!))
            copyto!(device, host)
            expected = f(input, dims)
            p = pf(x, args...; flags=FFTW.ESTIMATE)
            @test view_matches(f(x, args...), expected)
            @test view_matches(p * x, expected)
            y, outside_y = strided_destination(T, size(x))
            @test mul!(y, p, x) === y
            @test view_matches(y, expected)
            @test all(==(1), Array(parent(y))[outside_y])
            @test Array(device) == host

            # Exercise all three in-place entry points using the same plan.
            q = pf!(x, args...; flags=FFTW.ESTIMATE)
            for apply in (v -> q * v, v -> f!(v, args...), v -> mul!(v, q, v))
                copyto!(device, host)
                @test apply(x) === x
                @test view_matches(x, expected)
                @test Array(device)[outside] == host[outside]
            end
        end
    end
end

@testset "real FFT views" begin
    layouts = (("column gaps", (1:4, :, 1), nothing),
               ("odd length", (2:6, :, 1), (1,)),
               ("external batches", (1:4, :, :), (2,)))
    @testset "$T, $name" for T in (Float16, Float32, Float64),
                            (name, indices, region) in layouts
        # Half precision requires power-of-two sizes and unit real-side strides.
        T == Float16 && name != "column gaps" && continue
        host = rand(T, 8, 8, 8)
        device = CuArray(host)
        x = view(device, indices...)
        input = Array(view(host, indices...))
        args = isnothing(region) ? () : (region,)
        dims = isnothing(region) ? ntuple(identity, ndims(x)) : region
        d = size(x, first(dims))
        cpu = T == Float16 ? Float32.(input) : input
        expected = rfft(cpu, dims)
        @test which(plan_rfft, (typeof(x),)).module === cuFFT
        p = plan_rfft(x, args...; flags=FFTW.ESTIMATE)
        @test view_matches(rfft(x, args...), expected)
        dense_spectrum = p * x
        @test view_matches(dense_spectrum, expected)
        @test view_matches(mul!(similar(dense_spectrum), p, x), expected)
        spectrum, outside_s = strided_destination(Complex{T}, size(expected))
        @test mul!(spectrum, p, x) === spectrum
        @test view_matches(spectrum, expected)
        @test all(==(1), Array(parent(spectrum))[outside_s])
        @test Array(device) == host

        before = Array(parent(spectrum))
        cpu_spectrum = T == Float16 ? ComplexF32.(Array(spectrum)) : Array(spectrum)
        y, outside_y = strided_destination(T, size(x))
        @test which(plan_brfft, (typeof(spectrum), Int)).module === cuFFT
        for (f, pf) in ((brfft, plan_brfft), (irfft, plan_irfft))
            reference = f(cpu_spectrum, d, dims)
            q = pf(spectrum, d, args...; flags=FFTW.PRESERVE_INPUT)
            @test view_matches(f(spectrum, d, args...), reference)
            @test view_matches(q * spectrum, reference)
            @test which(mul!, (typeof(y), typeof(q), typeof(spectrum))).module === cuFFT
            @test mul!(y, q, spectrum) === y
            @test view_matches(y, reference)
            @test view_matches(mul!(similar(x), q, spectrum), reference)
            @test all(==(1), Array(parent(y))[outside_y])
        end
        @test view_matches(inv(p) * spectrum, input)
        @test view_matches(p \ spectrum, input)
        @test ldiv!(y, p, spectrum) === y
        @test view_matches(y, input)
        @test Array(parent(spectrum)) == before
    end
end

@testset "view validation and aliasing" begin
    host = rand(ComplexF32, 8, 4)
    device = CuArray(host)
    x = view(device, 1:4, :)
    y, _ = strided_destination(ComplexF32, size(x))
    p = plan_fft(x)
    @test_throws ArgumentError p * view(x, 1:2, :)
    @test_throws ArgumentError p * reshape(copy(x), length(x))
    @test_throws ArgumentError mul!(view(y, 1:2, :), p, x)
    @test_throws ArgumentError plan_fft(x, (1, 1))
    @test_throws ArgumentError mul!(x, p, x)
    @test_throws ArgumentError mul!(y, plan_fft!(x), x)
    @test_throws ArgumentError mul!(view(device, 1:2:8, :), plan_fft!(x), x)
    @test Array(device) == host
    @test all(==(1), Array(parent(y)))
    @test view_matches(mul!(similar(x), p, x), fft(host[1:4, :]))
    @test view_matches(plan_fft(copy(x)) * view(device, 1:2:8, :), fft(host[1:2:8, :]))

    # Disjoint and partially overlapping views of one parent.
    x, y = view(device, 1:2:8, :), view(device, 2:2:8, :)
    @test view_matches(mul!(y, plan_fft(x), x), fft(host[1:2:8, :]))
    @test Array(x) == host[1:2:8, :]
    host = rand(ComplexF32, 4, 5)
    device = CuArray(host)
    x, y = view(device, :, 1:4), view(device, :, 2:5)
    @test view_matches(mul!(y, plan_fft(x), x), fft(host[:, 1:4]))
    @test Array(device)[:, 1] == host[:, 1]
end
