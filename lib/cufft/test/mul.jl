# Create the same logical data in different layouts, with sentinels outside views.
function mul_storage(data, layout)
    layout == :dense && return (CuArray(data), data, falses(size(data)))
    dims = (2size(data, 1), size(data)[2:end]...)
    host = fill(eltype(data)(-17), dims)
    rows = layout == :gapped ? (1:size(data, 1)) :
           layout == :reversed ? (dims[1]:-2:2) : (2:2:dims[1])
    indices = (rows, ntuple(_ -> Colon(), ndims(data)-1)...)
    view(host, indices...) .= data
    outside = trues(dims)
    view(outside, indices...) .= false
    device = CuArray(host)
    view(device, indices...), host, outside
end
mul_parent(x) = x isa SubArray ? parent(x) : x

@testset "mul! with complex views" begin
    CUDACore.allowscalar(false)
    layouts = ((:dense, :dense), (:dense, :strided), (:gapped, :dense),
               (:strided, :gapped), (:reversed, :gapped))
    @testset "$T" for T in (ComplexF16, ComplexF32, ComplexF64)
        @testset "region=$dims" for (shape, dims) in (((4, 4), (1, 2)), ((4, 4, 8), (2,)))
            input = rand(T, shape)
            for (transform, planner) in ((fft, plan_fft), (bfft, plan_bfft), (ifft, plan_ifft))
                expected = transform(input, dims)
                for (input_layout, output_layout) in layouts
                    x, before_x, _ = mul_storage(input, input_layout)
                    y, before_y, outside = mul_storage(zero.(expected), output_layout)
                    p = planner(x, dims; flags=FFTW.ESTIMATE)
                    @test which(mul!, (typeof(y), typeof(p), typeof(x))).module === cuFFT
                    @test mul!(y, p, x) === y
                    @test isapprox(Array(y), expected; rtol=rtol(T), atol=atol(T))
                    @test Array(mul_parent(x)) == before_x
                    @test Array(mul_parent(y))[outside] == before_y[outside]
                end
            end
        end

        # In-place plans also accept distinct view objects with the same layout.
        input = rand(T, 4, 4)
        for layout in (:dense, :strided)
            x, before, outside = mul_storage(input, layout)
            y = layout == :dense ? x : view(parent(x), 2:2:size(parent(x), 1), :)
            p = plan_fft!(x)
            @test mul!(y, p, x) === y
            @test isapprox(Array(y), fft(input); rtol=rtol(T), atol=atol(T))
            @test mul!(x, inv(p), y) === x
            @test isapprox(Array(x), input; rtol=rtol(T), atol=atol(T))
            @test Array(mul_parent(x))[outside] == before[outside]
        end
    end
end

@testset "mul! with real transforms" begin
    layouts = ((:dense, :dense), (:dense, :strided), (:gapped, :dense),
               (:reversed, :gapped))
    @testset "$T" for T in (Float16, Float32, Float64)
        for (shape, dims) in (((4, 4), (1, 2)), ((4, 4, 8), (2,)))
            # Half precision does not support non-unit strides on the real side.
            T == Float16 && dims == (2,) && continue
            input = rand(T, shape)
            cpu = T == Float16 ? Float32.(input) : input
            expected = rfft(cpu, dims)
            d = size(input, first(dims))
            for (input_layout, output_layout) in layouts
                x, before_x, _ = mul_storage(input, input_layout)
                y, before_y, outside_y = mul_storage(Complex{T}.(zero.(expected)), output_layout)
                p = plan_rfft(x, dims)
                @test mul!(y, p, x) === y
                @test isapprox(Array(y), expected; rtol=rtol(T), atol=atol(T))
                @test Array(mul_parent(x)) == before_x
                @test Array(mul_parent(y))[outside_y] == before_y[outside_y]
                spectrum_before = Array(mul_parent(y))
                spectrum = T == Float16 ? ComplexF32.(Array(y)) : Array(y)
                for (q, reference) in ((plan_brfft(y, d, dims), brfft(spectrum, d, dims)),
                                       (plan_irfft(y, d, dims), irfft(spectrum, d, dims)),
                                       (inv(p), irfft(spectrum, d, dims)))
                    z, before_z, outside_z = mul_storage(zero.(input), input_layout)
                    @test which(mul!, (typeof(z), typeof(q), typeof(y))).module === cuFFT
                    @test mul!(z, q, y) === z
                    @test isapprox(Array(z), reference; rtol=rtol(T), atol=atol(T))
                    @test Array(mul_parent(y)) == spectrum_before
                    @test Array(mul_parent(z))[outside_z] == before_z[outside_z]
                end
                z, before_z, outside_z = mul_storage(zero.(input), input_layout)
                @test ldiv!(z, p, y) === z
                @test isapprox(Array(z), irfft(spectrum, d, dims); rtol=rtol(T), atol=atol(T))
                @test Array(mul_parent(y)) == spectrum_before
                @test Array(mul_parent(z))[outside_z] == before_z[outside_z]
            end
        end
    end
end

@testset "mul! validation and aliasing" begin
    host = rand(ComplexF32, 8, 4)
    device = CuArray(host)
    x = view(device, 1:4, :)
    p = plan_fft(x)
    @test_throws ArgumentError mul!(x, p, x)
    @test_throws ArgumentError mul!(x, plan_ifft(x), x)
    @test_throws ArgumentError mul!(copy(x), plan_fft!(x), x)
    @test_throws ArgumentError mul!(view(device, 1:2:8, :), plan_fft!(x), x)
    @test Array(device) == host

    y, before, _ = mul_storage(zeros(ComplexF32, 4, 4), :strided)
    @test_throws ArgumentError mul!(view(y, 1:2, :), p, x)
    @test_throws ArgumentError mul!(y, p, view(x, 1:2, :))
    @test Array(mul_parent(y)) == before
    @test Array(device) == host

    # Disjoint views of one parent are valid out-of-place destinations.
    x, y = view(device, 1:2:8, :), view(device, 2:2:8, :)
    @test mul!(y, plan_fft(x), x) === y
    @test Array(y) ≈ fft(host[1:2:8, :])
    @test Array(x) == host[1:2:8, :]

    # Snapshot overlapping dense input slices before writing the output.
    host = rand(ComplexF32, 4, 5)
    device = CuArray(host)
    x, y = view(device, :, 1:4), view(device, :, 2:5)
    expected = fft(host[:, 1:4])
    @test mul!(y, plan_fft(x), x) === y
    @test Array(y) ≈ expected
    @test Array(device)[:, 1] == host[:, 1]

    # Invalid shapes must not modify a C2R protection buffer or the output.
    spectrum = rfft(rand(Float32, 4, 4))
    x, before_x, _ = mul_storage(spectrum, :strided)
    y, before_y, _ = mul_storage(zeros(Float32, 4, 4), :gapped)
    p = plan_brfft(x, 4)
    fill!(p.buffer, ComplexF32(7))
    before_buffer = Array(p.buffer)
    @test_throws ArgumentError mul!(view(y, 1:2, :), p, x)
    @test_throws ArgumentError mul!(y, p, reshape(copy(x), length(x)))
    @test Array(p.buffer) == before_buffer
    @test Array(mul_parent(x)) == before_x
    @test Array(mul_parent(y)) == before_y
end
