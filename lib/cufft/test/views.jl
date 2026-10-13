@testset "out-of-place complex FFTs of strided views" begin
    CUDACore.allowscalar(false)
    @testset "$T" for T in (ComplexF16, ComplexF32, ComplexF64)
        host = rand(T, 8, T == ComplexF16 ? 8 : 6)
        device = CuArray(host)
        layouts = (
            ("gaps between columns", (1:4, :)),
            ("single row", (2, :)),
            ("stepped rows", (1:2:8, :)),
            ("offset submatrix", (2:5, 3:6)),
        )

        @testset "$name" for (name, indices) in layouts
            cpu_view = view(host, indices...)
            cpu_input = Array(cpu_view)
            gpu_view = view(device, indices...)
            @test gpu_view isa SubArray
            @test gpu_view isa StridedCuArray

            @test which(plan_fft, (typeof(gpu_view),)).module === cuFFT
            @test which(plan_bfft, (typeof(gpu_view),)).module === cuFFT
            @test isapprox(Array(fft(gpu_view)), fft(cpu_input);
                           rtol=rtol(T), atol=atol(T))

            regions = ndims(gpu_view) == 1 ? (1, (1,), 1:1) :
                                            (1, 2, (1, 2), (2, 1), 1:2)
            @testset "region=$region" for region in regions
                for (transform, planner) in ((fft, plan_fft),
                                             (bfft, plan_bfft),
                                             (ifft, plan_ifft))
                    expected = transform(cpu_input, region)
                    plan = planner(gpu_view, region; flags=FFTW.ESTIMATE)
                    result = plan * gpu_view
                    @test result isa CuArray{T}
                    @test isapprox(Array(result), expected;
                                   rtol=rtol(T), atol=atol(T))

                    dense_plan = planner(copy(gpu_view), region)
                    @test isapprox(Array(dense_plan * gpu_view), expected;
                                   rtol=rtol(T), atol=atol(T))
                end
            end

            @test Array(device) == host
        end

        # Dense staging allows one plan to be reused across different strides.
        first_view = view(device, 1:4, :)
        second_view = view(device, 1:2:8, :)
        plan = plan_fft(first_view; flags=FFTW.ESTIMATE)
        @test isapprox(Array(plan * second_view), fft(host[1:2:8, :]);
                       rtol=rtol(T), atol=atol(T))
        @test_throws ArgumentError plan * view(device, 1:2, :)
        @test_throws ArgumentError plan_fft(first_view, (1, 1))
        @test Array(device) == host
    end
end

@testset "in-place complex FFTs of strided views" begin
    CUDACore.allowscalar(false)
    @testset "$T" for T in (ComplexF16, ComplexF32, ComplexF64)
        host = rand(T, 8, 8, 4)
        layouts = (
            ("row", (2, :, 2), nothing),
            ("stepped offset", (2:2:8, 2:5, 2), nothing),
            ("contiguous", (:, :, 2), nothing),
            ("external batches", (1:4, :, :), (2,)),
        )
        @testset "$name" for (name, indices, region) in layouts
            outside = trues(size(host))
            view(outside, indices...) .= false
            input = Array(view(host, indices...))
            dims = isnothing(region) ? ntuple(identity, ndims(input)) : region
            for (transform, planner) in ((fft, plan_fft!), (bfft, plan_bfft!),
                                         (ifft, plan_ifft!))
                device = CuArray(host)
                v = view(device, indices...)
                @test which(plan_fft!, (typeof(v),)).module === cuFFT
                @test which(plan_bfft!, (typeof(v),)).module === cuFFT
                plan = isnothing(region) ? planner(v; flags=FFTW.ESTIMATE) :
                                           planner(v, region; flags=FFTW.ESTIMATE)
                @test plan * v === v
                @test isapprox(Array(v), transform(input, dims);
                               rtol=rtol(T), atol=atol(T))
                @test Array(device)[outside] == host[outside]
            end
        end

        # Reuse a dense-layout plan on a different stride, then invert it.
        matrix = host[:, :, 1]
        device = CuArray(matrix)
        plan = plan_fft!(CuArray(matrix[1:4, :]))
        v = view(device, 1:2:8, :)
        @test plan * v === v
        @test inv(plan) * v === v
        @test isapprox(Array(device), matrix; rtol=rtol(T), atol=atol(T))
        before = Array(device)
        @test_throws ArgumentError plan * view(device, 1:2, :)
        @test Array(device) == before
        @test_throws ArgumentError plan_fft!(v, (1, 1))
    end
end
