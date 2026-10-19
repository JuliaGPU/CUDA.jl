# a strided view into a parent filled with ones, and a mask of the elements outside it
function strided_destination(T, dims)
    storage = CuArray(ones(T, (2dims[1], dims[2:end]...)))
    indices = (2:2:size(storage, 1), ntuple(_ -> Colon(), length(dims)-1)...)
    outside = trues(size(storage))
    view(outside, indices...) .= false
    view(storage, indices...), outside
end

function out_of_place_view(X::AbstractArray{T}, indices, region...) where {T <: Complex}
    v = X[indices...]
    d_X = CuArray(X)
    d_v = view(d_X, indices...)
    d_Y, outside = strided_destination(T, size(v))

    @inferred plan_fft(d_v, region...)
    for (f, planner) in ((fft, plan_fft), (ifft, plan_ifft))
        fftw_X = f(v, region...)
        p = planner(d_v, region...)

        @test isapprox(collect(f(d_v, region...)), fftw_X, rtol = rtol(T), atol = atol(T))
        @test isapprox(collect(p * d_v), fftw_X, rtol = rtol(T), atol = atol(T))

        @test mul!(d_Y, p, d_v) === d_Y
        @test isapprox(collect(d_Y), fftw_X, rtol = rtol(T), atol = atol(T))
        @test all(==(1), Array(parent(d_Y))[outside])
        @test Array(d_X) == X
    end
end

function in_place_view(X::AbstractArray{T}, indices, region...) where {T <: Complex}
    v = X[indices...]
    d_X = CuArray(X)
    d_v = view(d_X, indices...)
    outside = trues(size(X))
    view(outside, indices...) .= false

    @inferred plan_fft!(d_v, region...)
    for (f!, planner) in ((fft!, plan_fft!), (ifft!, plan_ifft!))
        fftw_X = f!(copy(v), region...)
        p = planner(d_v, region...)

        @test p * d_v === d_v
        @test isapprox(collect(d_v), fftw_X, rtol = rtol(T), atol = atol(T))
        @test Array(d_X)[outside] == X[outside]

        copyto!(d_X, X)
        @test f!(d_v, region...) === d_v
        @test isapprox(collect(d_v), fftw_X, rtol = rtol(T), atol = atol(T))
        @test Array(d_X)[outside] == X[outside]

        copyto!(d_X, X)
        @test mul!(d_v, p, d_v) === d_v
        @test isapprox(collect(d_v), fftw_X, rtol = rtol(T), atol = atol(T))
        @test Array(d_X)[outside] == X[outside]

        copyto!(d_X, X)
    end
end

function out_of_place_view(X::AbstractArray{T}, indices, region...) where {T <: Real}
    v = X[indices...]
    d_X = CuArray(X)
    d_v = view(d_X, indices...)
    d = size(v, isempty(region) ? 1 : first(only(region)))

    fftw_Y = rfft(v, region...)
    p = @inferred plan_rfft(d_v, region...)
    @test isapprox(collect(rfft(d_v, region...)), fftw_Y, rtol = rtol(T), atol = atol(T))
    @test isapprox(collect(p * d_v), fftw_Y, rtol = rtol(T), atol = atol(T))
    d_Y, outside_Y = strided_destination(Complex{T}, size(fftw_Y))
    @test mul!(d_Y, p, d_v) === d_Y
    @test isapprox(collect(d_Y), fftw_Y, rtol = rtol(T), atol = atol(T))
    @test all(==(1), Array(parent(d_Y))[outside_Y])
    @test Array(d_X) == X

    # complex-to-real transforms of a strided spectrum, which must be preserved
    Y = collect(d_Y)
    parent_Y = Array(parent(d_Y))
    d_Z, outside_Z = strided_destination(T, size(v))
    for (f, planner) in ((brfft, plan_brfft), (irfft, plan_irfft))
        fftw_Z = f(Y, d, region...)
        pinv = planner(d_Y, d, region...)
        @test isapprox(collect(f(d_Y, d, region...)), fftw_Z, rtol = rtol(T), atol = atol(T))
        @test isapprox(collect(pinv * d_Y), fftw_Z, rtol = rtol(T), atol = atol(T))
        @test mul!(d_Z, pinv, d_Y) === d_Z
        @test isapprox(collect(d_Z), fftw_Z, rtol = rtol(T), atol = atol(T))
        @test all(==(1), Array(parent(d_Z))[outside_Z])
    end
    @test ldiv!(d_Z, p, d_Y) === d_Z
    @test isapprox(collect(d_Z), v, rtol = rtol(T), atol = atol(T))
    @test Array(parent(d_Y)) == parent_Y
end

# staging views does not depend on the element type, so only test one of each kind
@testset "complex" begin
    X = rand(ComplexF32, 8, 8, 4)
    @testset "$name" for (name, indices) in (("row", (2, :, 1)),
                                             ("reversed", (8:-2:2, 2:5, 1)))
        out_of_place_view(X, indices)
        in_place_view(X, indices)
    end
    @testset "external batches" begin
        out_of_place_view(X, (1:4, :, :), (2,))
        in_place_view(X, (1:4, :, :), (2,))
    end
end

@testset "real" begin
    X = rand(Float32, 8, 8, 8)
    @testset "columns" begin
        out_of_place_view(X, (1:4, :, 1))
    end
    @testset "odd length" begin
        out_of_place_view(X, (2:6, :, 1), (1,))
    end
    # an odd batch stride, so that half of the batches are not complex-aligned
    @testset "external batches" begin
        out_of_place_view(X, (1:3, 1:3, :), (2,))
    end
end

@testset "validation and aliasing" begin
    T = ComplexF32
    X = rand(T, 8, 4)
    d_X = CuArray(X)
    d_v = view(d_X, 1:4, :)
    d_Y = CuArray(ones(T, 8, 4))
    d_w = view(d_Y, 2:2:8, :)
    p = plan_fft(d_v)

    @test_throws ArgumentError p * view(d_v, 1:2, :)
    @test_throws ArgumentError mul!(view(d_w, 1:2, :), p, d_v)
    @test_throws ArgumentError mul!(d_v, p, d_v)
    @test_throws ArgumentError mul!(d_w, plan_fft!(d_v), d_v)
    @test_throws ArgumentError mul!(view(d_X, 1:2:8, :), plan_fft!(d_v), d_v)
    @test Array(d_X) == X
    @test all(==(1), Array(d_Y))

    # a plan only depends on the size, not the strides
    p_dense = plan_fft(copy(d_v))
    @test isapprox(collect(p_dense * view(d_X, 1:2:8, :)), fft(X[1:2:8, :]), rtol = rtol(T), atol = atol(T))

    # disjoint views of the same parent
    d_v, d_w = view(d_X, 1:2:8, :), view(d_X, 2:2:8, :)
    mul!(d_w, plan_fft(d_v), d_v)
    @test isapprox(collect(d_w), fft(X[1:2:8, :]), rtol = rtol(T), atol = atol(T))
    @test Array(d_v) == X[1:2:8, :]

    # partially overlapping views of the same parent
    X = rand(T, 4, 5)
    d_X = CuArray(X)
    d_v, d_w = view(d_X, :, 1:4), view(d_X, :, 2:5)
    mul!(d_w, plan_fft(d_v), d_v)
    @test isapprox(collect(d_w), fft(X[:, 1:4]), rtol = rtol(T), atol = atol(T))
    @test Array(d_X)[:, 1] == X[:, 1]
end
