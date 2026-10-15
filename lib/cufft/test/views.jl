# Create a strided output view, leaving unused rows at one.
function strided_destination(T, dims)
    storage = CuArray(ones(T, (2dims[1], dims[2:end]...)))
    indices = (2:2:size(storage, 1), ntuple(_ -> Colon(), length(dims)-1)...)
    outside = trues(size(storage))
    view(outside, indices...) .= false
    view(storage, indices...), outside
end

function out_of_place_view(X::AbstractArray{T}, indices, region...) where {T <: Complex}
    v = Array(view(X, indices...))
    d_X = CuArray(X)
    d_v = view(d_X, indices...)
    d_Y, outside = strided_destination(T, size(v))

    @test which(plan_fft, (typeof(d_v),)).module === cuFFT
    @test which(plan_bfft, (typeof(d_v),)).module === cuFFT
    for (f, planner) in ((fft, plan_fft), (bfft, plan_bfft), (ifft, plan_ifft))
        fftw_X = f(v, region...)
        p = planner(d_v, region...; flags=FFTW.ESTIMATE)

        @test isapprox(Array(f(d_v, region...)), fftw_X, rtol = rtol(T), atol = atol(T))
        @test isapprox(Array(p * d_v), fftw_X, rtol = rtol(T), atol = atol(T))

        @test mul!(d_Y, p, d_v) === d_Y
        @test isapprox(Array(d_Y), fftw_X, rtol = rtol(T), atol = atol(T))
        @test all(==(1), Array(parent(d_Y))[outside])
        @test Array(d_X) == X
    end
end

function in_place_view(X::AbstractArray{T}, indices, region...) where {T <: Complex}
    v = Array(view(X, indices...))
    d_X = CuArray(X)
    d_v = view(d_X, indices...)
    outside = trues(size(X))
    view(outside, indices...) .= false

    @test which(plan_fft!, (typeof(d_v),)).module === cuFFT
    @test which(plan_bfft!, (typeof(d_v),)).module === cuFFT
    for (f!, planner) in ((fft!, plan_fft!), (bfft!, plan_bfft!), (ifft!, plan_ifft!))
        fftw_X = f!(copy(v), region...)
        copyto!(d_X, X)
        p = planner(d_v, region...; flags=FFTW.ESTIMATE)

        # Apply an in-place plan.
        @test p * d_v === d_v
        @test isapprox(Array(d_v), fftw_X, rtol = rtol(T), atol = atol(T))
        @test Array(d_X)[outside] == X[outside]

        # Call the in-place transform directly.
        copyto!(d_X, X)
        @test f!(d_v, region...) === d_v
        @test isapprox(Array(d_v), fftw_X, rtol = rtol(T), atol = atol(T))
        @test Array(d_X)[outside] == X[outside]

        # Use the same view as input and output.
        copyto!(d_X, X)
        @test mul!(d_v, p, d_v) === d_v
        @test isapprox(Array(d_v), fftw_X, rtol = rtol(T), atol = atol(T))
        @test Array(d_X)[outside] == X[outside]
    end
end

function out_of_place_view(X::AbstractArray{T}, indices, region...) where {T <: Real}
    v = Array(view(X, indices...))
    d_X = CuArray(X)
    d_v = view(d_X, indices...)
    dims = isempty(region) ? (1:ndims(v)) : only(region)
    d = size(v, first(dims))
    # Use Float32 CPU references for half precision.
    cpu = T == Float16 ? Float32.(v) : v
    fftw_X = rfft(cpu, region...)

    @test which(plan_rfft, (typeof(d_v),)).module === cuFFT
    p = plan_rfft(d_v, region...; flags=FFTW.ESTIMATE)
    @test isapprox(Array(rfft(d_v, region...)), fftw_X, rtol = rtol(T), atol = atol(T))
    d_spectrum = p * d_v
    @test isapprox(Array(d_spectrum), fftw_X, rtol = rtol(T), atol = atol(T))

    # Write R2C results into dense and strided outputs.
    d_result = similar(d_spectrum)
    mul!(d_result, p, d_v)
    @test isapprox(Array(d_result), fftw_X, rtol = rtol(T), atol = atol(T))
    d_Y, outside_Y = strided_destination(Complex{T}, size(d_spectrum))
    @test mul!(d_Y, p, d_v) === d_Y
    @test isapprox(Array(d_Y), fftw_X, rtol = rtol(T), atol = atol(T))
    @test all(==(1), Array(parent(d_Y))[outside_Y])
    @test Array(d_X) == X

    # C2R transforms must preserve the spectrum, including when using mul!.
    spectrum_before = Array(parent(d_Y))
    cpu_spectrum = T == Float16 ? ComplexF32.(Array(d_Y)) : Array(d_Y)
    d_Z, outside_Z = strided_destination(T, size(v))
    @test which(plan_brfft, (typeof(d_Y), Int)).module === cuFFT
    for (f, planner) in ((brfft, plan_brfft), (irfft, plan_irfft))
        fftw_Y = f(cpu_spectrum, d, region...)
        pinv = planner(d_Y, d, region...; flags=FFTW.PRESERVE_INPUT)
        @test isapprox(Array(f(d_Y, d, region...)), fftw_Y, rtol = rtol(T), atol = atol(T))
        @test isapprox(Array(pinv * d_Y), fftw_Y, rtol = rtol(T), atol = atol(T))

        @test which(mul!, (typeof(d_Z), typeof(pinv), typeof(d_Y))).module === cuFFT
        @test mul!(d_Z, pinv, d_Y) === d_Z
        @test isapprox(Array(d_Z), fftw_Y, rtol = rtol(T), atol = atol(T))
        d_result = similar(d_v)
        mul!(d_result, pinv, d_Y)
        @test isapprox(Array(d_result), fftw_Y, rtol = rtol(T), atol = atol(T))
        @test all(==(1), Array(parent(d_Z))[outside_Z])
    end

    @test isapprox(Array(inv(p) * d_Y), v, rtol = rtol(T), atol = atol(T))
    @test isapprox(Array(p \ d_Y), v, rtol = rtol(T), atol = atol(T))
    @test ldiv!(d_Z, p, d_Y) === d_Z
    @test isapprox(Array(d_Z), v, rtol = rtol(T), atol = atol(T))
    @test Array(parent(d_Y)) == spectrum_before
end

CUDACore.allowscalar(false)

@testset "complex views ($T)" for T in [ComplexF16, ComplexF32, ComplexF64]
    X = rand(T, 8, 8, 4)
    @testset "$name" for (name, indices) in (
        ("row", (2, :, 1)),
        ("reversed offset", (8:-2:2, 2:5, 1)),
        ("contiguous", (:, :, 1)),
    )
        @testset "out-of-place" begin
            out_of_place_view(X, indices)
        end
        @testset "in-place" begin
            in_place_view(X, indices)
        end
    end
    @testset "external batches" begin
        @testset "out-of-place" begin
            out_of_place_view(X, (1:4, :, :), (2,))
        end
        @testset "in-place" begin
            in_place_view(X, (1:4, :, :), (2,))
        end
    end
end

@testset "real views ($T)" for T in [Float16, Float32, Float64]
    X = rand(T, 8, 8, 8)
    @testset "column gaps" begin
        out_of_place_view(X, (1:4, :, 1))
    end
    # Half precision requires power-of-two sizes and unit real-side strides.
    if T != Float16
        @testset "odd length" begin
            out_of_place_view(X, (2:6, :, 1), (1,))
        end
        @testset "external batches" begin
            out_of_place_view(X, (1:4, :, :), (2,))
        end
    end
end

@testset "view validation and aliasing" begin
    T = ComplexF32
    X = rand(T, 8, 4)
    d_X = CuArray(X)
    d_v = view(d_X, 1:4, :)
    d_Y = CuArray(ones(T, 8, 4))
    d_w = view(d_Y, 2:2:8, :)
    p = plan_fft(d_v)

    # Reject invalid shapes, dimensions, and input/output layouts without modifying data.
    @test_throws ArgumentError p * view(d_v, 1:2, :)
    @test_throws ArgumentError p * reshape(copy(d_v), length(d_v))
    @test_throws ArgumentError mul!(view(d_w, 1:2, :), p, d_v)
    @test_throws ArgumentError plan_fft(d_v, (1, 1))
    @test_throws ArgumentError mul!(d_v, p, d_v)
    @test_throws ArgumentError mul!(d_w, plan_fft!(d_v), d_v)
    @test_throws ArgumentError mul!(view(d_X, 1:2:8, :), plan_fft!(d_v), d_v)
    @test Array(d_X) == X
    @test all(==(1), Array(d_Y))

    # Support dense outputs on strided inputs.
    d_result = similar(d_v)
    mul!(d_result, p, d_v)
    @test isapprox(Array(d_result), fft(X[1:4, :]), rtol = rtol(T), atol = atol(T))

    # Reuse a dense plan on a view with the same shape but different strides.
    p_dense = plan_fft(copy(d_v))
    d_other = view(d_X, 1:2:8, :)
    @test isapprox(Array(p_dense * d_other), fft(X[1:2:8, :]), rtol = rtol(T), atol = atol(T))

    # Disjoint views of the same parent.
    d_v, d_w = view(d_X, 1:2:8, :), view(d_X, 2:2:8, :)
    mul!(d_w, plan_fft(d_v), d_v)
    @test isapprox(Array(d_w), fft(X[1:2:8, :]), rtol = rtol(T), atol = atol(T))
    @test Array(d_v) == X[1:2:8, :]

    # Partially overlapping views of the same parent.
    X = rand(T, 4, 5)
    d_X = CuArray(X)
    d_v, d_w = view(d_X, :, 1:4), view(d_X, :, 2:5)
    mul!(d_w, plan_fft(d_v), d_v)
    @test isapprox(Array(d_w), fft(X[:, 1:4]), rtol = rtol(T), atol = atol(T))
    @test Array(d_X)[:, 1] == X[:, 1]
end
