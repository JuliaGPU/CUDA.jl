@testset "library properties" begin
    # these identify the toolkit the library was built with, not `version()`
    @test cuStateVec.property(CUDACore.MAJOR_VERSION) > 0
    @test cuStateVec.property(CUDACore.MINOR_VERSION) >= 0
    @test cuStateVec.property(CUDACore.PATCH_LEVEL) >= 0
    @test cuStateVec.currentStream() == CUDACore.stream().handle
    @test cuStateVec.errorName(cuStateVec.CUSTATEVEC_STATUS_INVALID_VALUE) == "CUSTATEVEC_STATUS_INVALID_VALUE"
    @test !isempty(cuStateVec.errorString(cuStateVec.CUSTATEVEC_STATUS_INVALID_VALUE))
end

@testset "streams" begin
    # the library follows the task's stream
    original = CUDACore.stream()
    other = CuStream()
    try
        CUDACore.stream!(other)
        @test cuStateVec.currentStream() == other.handle
    finally
        CUDACore.stream!(original)
    end
    @test cuStateVec.currentStream() == original.handle
end

@testset "compute types" begin
    @test_throws ArgumentError convert(cuStateVec.custatevecComputeType_t, Int32)
end

@testset "JULIA_DEBUG" begin
    script = """
        using CUDACore, cuStateVec, Logging
        io = IOBuffer()
        with_logger(SimpleLogger(io, Logging.Debug)) do
            cuStateVec.CuStateVec(ComplexF64, 2)
            CUDACore.flush_logs()
        end
        @assert occursin("custatevec", String(take!(io)))
        """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    @test success(pipeline(addenv(cmd, "JULIA_DEBUG" => "cuStateVec"); stdout=devnull, stderr))
end

@testset "math mode" begin
    original = cuStateVec.mathMode()
    @test original == cuStateVec.CUSTATEVEC_MATH_MODE_DEFAULT
    try
        for mode in (cuStateVec.CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9,
                     cuStateVec.CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9)
            @test cuStateVec.mathMode!(mode) == mode
            @test cuStateVec.mathMode() == mode
            # the mode must not break regular operation
            sv = CuStateVec(ComplexF32, 2)
            applyMatrix!(sv, ComplexF32[0 1; 1 0], false, [0], Int[])
            @test collect(sv.data) ≈ ComplexF32[0, 1, 0, 0]
        end
    finally
        cuStateVec.mathMode!(original)
    end
end

@testset "workspace" begin
    @test cuStateVec.defaultWorkspaceSize() > 0
end

@testset "sampler" begin
    @testset for elty in [ComplexF32, ComplexF64]
        # |ψ⟩ = (|00⟩ + |11⟩)/√2, scaled to have squared norm 2
        sv = CuStateVec(elty[1, 0, 0, 1])
        sampler = cuStateVec.CuStateVecSampler(sv, UInt32(16))
        cuStateVec.preprocess!(sampler)
        @test cuStateVec.squaredNorm(sampler) ≈ 2

        # with sorted random numbers, the first half lands in |00⟩, and the second in |11⟩
        randnums = [0.1, 0.2, 0.7, 0.9]
        @test sample(sampler, [0, 1], randnums) == [0, 0, 3, 3]
        @test sample(sampler, [1], randnums) == [0, 0, 1, 1]
        @test sample(sampler, [0, 1], randnums; output=cuStateVec.CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER) == [0, 0, 3, 3]
    end

    @testset "sub state vectors" begin
        # two halves of (|000⟩ + |001⟩ + |100⟩ + |101⟩)/2: the top bit selects the part
        parts = [ComplexF64[0.5, 0.5, 0, 0], ComplexF64[0.5, 0.5, 0, 0]]
        samplers = map(parts) do part
            cuStateVec.CuStateVecSampler(CuStateVec(part), UInt32(8))
        end
        foreach(cuStateVec.preprocess!, samplers)
        norms = cuStateVec.squaredNorm.(samplers)
        @test norms ≈ [0.5, 0.5]
        cuStateVec.applySubSVOffset!(samplers[1], 0, 2, 0.0, sum(norms))
        cuStateVec.applySubSVOffset!(samplers[2], 1, 2, norms[1], sum(norms))
        @test sample(samplers[1], [0], [0.1, 0.4]) == [0, 1]
        @test sample(samplers[2], [0], [0.6, 0.9]) == [0, 1]

        @test_throws cuStateVec.CUSTATEVECError cuStateVec.applySubSVOffset!(samplers[1], 2, 2, 0.0, 1.0)
    end
end

@testset "accessor" begin
    @testset for elty in [ComplexF32, ComplexF64]
        data = elty[1, 2, 3, 4, 5, 6, 7, 8]

        @testset "host and device buffers" begin
            sv = CuStateVec(copy(data))
            acc = CuStateVecAccessor(sv, [0, 1, 2], Int[], Int[])
            h_buf = similar(data)
            accessorGet(acc, h_buf, 0, 8)
            @test h_buf == data
            d_buf = CuVector{elty}(undef, 8)
            accessorGet(acc, d_buf, 0, 8)
            @test collect(d_buf) == data

            accessorSet!(acc, CuVector(reverse(data)), 0, 8)
            @test collect(sv.data) == reverse(data)
        end

        @testset "views" begin
            sv = CuStateVec(copy(data))
            acc = CuStateVecAccessor(sv, [0, 1, 2], Int[], Int[]; view=true)
            buf = similar(data)
            accessorGet(acc, buf, 0, 8)
            @test buf == data
            @test_throws cuStateVec.CUSTATEVECError accessorSet!(acc, buf, 0, 8)
        end

        @testset "extra workspace" begin
            sv = CuStateVec(copy(data))
            acc = CuStateVecAccessor(sv, [2, 1, 0], Int[], Int[])
            # a workspace is provided automatically when one is needed
            @test (acc.workspace !== nothing) == (acc.ws_size > 0)
            buf = similar(data)
            accessorGet(acc, buf, 0, 8)
            # the bit ordering is reversed
            @test buf == data[[1, 5, 3, 7, 2, 6, 4, 8]]

            if acc.ws_size > 0
                @test_throws ArgumentError cuStateVec.setExtraWorkspace!(acc, CuVector{UInt8}(undef, acc.ws_size - 1))
            end
            workspace = CuVector{UInt8}(undef, acc.ws_size + 16)
            @test cuStateVec.setExtraWorkspace!(acc, workspace) === acc
            @test acc.workspace === workspace
            accessorGet(acc, buf, 0, 8)
            @test buf == data[[1, 5, 3, 7, 2, 6, 4, 8]]
        end
    end
end

@testset "integer vectors of any width" begin
    # the C API takes `int32_t` arrays: wider integers must be converted, not reinterpreted
    @testset for T in (Int32, Int64)
        data = ComplexF64[1, 0, 0, 0, 0, 0, 0, 1] ./ sqrt(2)
        @test collect(abs2SumOnZBasis(CuStateVec(copy(data)), T[2])) ≈ [0.5, 0.5]
        # all of the amplitude has odd parity on the top bit when collapsing to |1⟩
        sv = CuStateVec(ComplexF64[0, 0, 0, 0, 0, 0, 0, 1])
        @test last(measureOnZBasis!(sv, T[2], 0.5)) == 1
        sv = CuStateVec(ComplexF64[0, 1, 0, 0, 0, 0, 0, 0])
        @test last(measureOnZBasis!(sv, T[2], 0.5)) == 0

        # masks select the part of the state vector with bit 2 set
        acc = CuStateVecAccessor(CuStateVec(ComplexF64[1, 2, 3, 4, 5, 6, 7, 8]), [0, 1], [1], [2])
        buf = zeros(ComplexF64, 4)
        accessorGet(acc, buf, 0, 4)
        @test buf == ComplexF64[5, 6, 7, 8]
    end

    # one matrix for each batched state vector, selected by index
    X = ComplexF64[0, 1, 1, 0]
    I2 = ComplexF64[1, 0, 0, 1]
    sv = CuStateVec(ComplexF64[1, 0, 1, 0])
    applyMatrixBatched!(sv, 2, cuStateVec.CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, [1, 0], vcat(X, I2), 2, false, [0], Int[])
    @test collect(sv.data) == ComplexF64[1, 0, 0, 1]
end

@testset "distributed index bit swap scheduler" begin
    # two global and two local bits: swapping a local with a global bit exchanges half of
    # each sub state vector with its counterpart. the first batch has nothing to transfer.
    scheduler = cuStateVec.CuStateVecDistIndexBitSwapScheduler(2, 2)
    @test cuStateVec.setIndexBitSwaps!(scheduler, [0 => 2]) == 2
    for org in 0:3
        params = cuStateVec.swapParameters(scheduler, 0, org)
        @test params.swapBatchIndex == 0
        @test params.orgSubSVIndex == org
        @test params.dstSubSVIndex == org
        @test params.transferSize == 0
        @test params.dataTransferType == cuStateVec.CUSTATEVEC_DATA_TRANSFER_TYPE_NONE

        params = cuStateVec.swapParameters(scheduler, 1, org)
        @test params.swapBatchIndex == 1
        @test params.orgSubSVIndex == org
        @test params.dstSubSVIndex == xor(org, 1)
        @test params.transferSize == 2
        @test params.nSegmentBits == 2
        @test params.segmentMaskLen == 1
        @test params.segmentMaskOrdering[1] == 0
        @test params.orgSegmentMaskString[1] == (org & 1 == 0 ? 1 : 0)
        @test params.dstSegmentMaskString[1] == (org & 1 == 0 ? 0 : 1)
        @test params.dataTransferType == cuStateVec.CUSTATEVEC_DATA_TRANSFER_TYPE_SEND_RECV
    end

    # two swaps exchange a quarter with each of the three other sub state vectors
    @test cuStateVec.setIndexBitSwaps!(scheduler, [0 => 2, 1 => 3]) == 4
    @test cuStateVec.swapParameters(scheduler, 3, 0).dstSubSVIndex == 3
    @test cuStateVec.swapParameters(scheduler, 3, 0).transferSize == 1

    # and masks restrict the elements that are exchanged
    n_batches = cuStateVec.setIndexBitSwaps!(scheduler, [0 => 2], [1], [1])
    @test n_batches >= 1
    @test cuStateVec.swapParameters(scheduler, n_batches - 1, 0).transferSize == 1

    @test_throws cuStateVec.CUSTATEVECError cuStateVec.swapParameters(scheduler, n_batches, 0)
end

@testset "sub state vector migrator" begin
    @testset for elty in [ComplexF32, ComplexF64]
        slices = CuVector{elty}(undef, 2 * 4)
        migrator = cuStateVec.CuStateVecSubSVMigrator(slices, 2, 2)
        a = CuVector(elty[1, 2, 3, 4])
        b = CuVector(elty[5, 6, 7, 8])

        # one way each
        cuStateVec.migrate!(migrator, 0, a, nothing, 0, 4)
        cuStateVec.migrate!(migrator, 1, b, nothing, 0, 4)
        CUDACore.synchronize()
        @test collect(slices) == elty[1, 2, 3, 4, 5, 6, 7, 8]
        out = CUDACore.zeros(elty, 4)
        cuStateVec.migrate!(migrator, 1, nothing, out, 0, 4)
        CUDACore.synchronize()
        @test collect(out) == elty[5, 6, 7, 8]

        # exchanging a part of the slice
        src = CuVector(elty[9, 10, 11, 12])
        dst = CUDACore.zeros(elty, 4)
        cuStateVec.migrate!(migrator, 0, src, dst, 1, 3)
        CUDACore.synchronize()
        @test collect(slices)[1:4] == elty[1, 10, 11, 4]
        @test collect(dst) == elty[0, 2, 3, 0]

        @test_throws DimensionMismatch cuStateVec.CuStateVecSubSVMigrator(slices, 3, 2)
    end
end
