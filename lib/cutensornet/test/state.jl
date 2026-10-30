using cuTensorNet: CuNetworkState, CuNetworkOperator, CuStateAccessor, CuStateExpectation,
    CuStateMarginal, CuStateSampler, CuStateProjectionMPS, apply!, apply_with_gradient!,
    update_gradient!, apply_diagonal!, apply_controlled!, update!, apply_unitary_channel!,
    apply_general_channel!, initialize_mps!, finalize_mps!, capture_mps!, details, configure!,
    info, prepare!, compute!, compute_with_gradients!, flops, marginal_kind, sample!,
    append_product!, append_mpo!, tensor_info, compute_env!, extract_tensor!, insert_tensor!,
    update_coefficient!, update_dual_tensors!

# the elements of gates are stored following the rows of their textbook definition
row_order(M, dims...) = reshape(permutedims(M), dims...)
hadamard(T) = CuArray(T[1 1; 1 -1] ./ sqrt(real(T)(2)))
pauli_x(T)  = CuArray(T[0 1; 1 0])
pauli_y(T)  = CuArray(T[0 -im; im 0])
pauli_z(T)  = CuArray(T[1 0; 0 -1])
cnot(T)     = CuArray(row_order(T[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0], 2, 2, 2, 2))

# (|00⟩ + |11⟩)/√2
function bell_state(T)
    state = CuNetworkState{T}([2, 2])
    apply!(state, [0], hadamard(T); unitary=true)
    apply!(state, [0, 1], cnot(T); unitary=true)
    return state
end

const BELL = [1 0; 0 1] ./ sqrt(2)

@testset "state vectors" begin
    @testset for T in [ComplexF32, ComplexF64]
        state = bell_state(T)
        @test eltype(state) == T
        @test_throws ArgumentError compute!(state)
        prepare!(state)
        @test flops(state) >= 0
        output = compute!(state)
        @test output.extents == [[2, 2]]
        @test only(output.tensors) isa CuArray{T,2}
        @test collect(only(output.tensors)) ≈ BELL

        d = details(state)
        @test d.extents == [[2, 2]]
        @test d.strides == [[1, 2]]

        @test_throws DimensionMismatch apply!(state, [0], cnot(T))
        @test_throws DimensionMismatch apply_with_gradient!(state, [0], cnot(T), CUDACore.zeros(T, 2, 2, 2, 2))
        @test_throws DimensionMismatch apply_with_gradient!(state, [0], hadamard(T), CUDACore.zeros(T, 2))
        @test_throws DimensionMismatch apply_diagonal!(state, [0], cnot(T))
        @test_throws DimensionMismatch apply_controlled!(state, [0], [1], cnot(T))
    end

    @testset "updating operators" begin
        T = ComplexF64
        state = CuNetworkState{T}([2])
        id = apply!(state, [0], pauli_x(T); immutable=false)
        prepare!(state)
        @test collect(only(compute!(state).tensors)) ≈ [0, 1]
        update!(state, id, CuArray(T[1 0; 0 1]))
        @test collect(only(compute!(state).tensors)) ≈ [1, 0]

        # changing whether it's unitary needs a new preparation
        update!(state, id, pauli_x(T); unitary=true)
        @test_throws cuTensorNet.CUTENSORNETError compute!(state)
        prepare!(state)
        @test collect(only(compute!(state).tensors)) ≈ [0, 1]
    end

    @testset "adjoints" begin
        T = ComplexF64
        # S† |1⟩ = -i |1⟩
        S = CuArray(T[1 0; 0 im])
        state = CuNetworkState{T}([2])
        apply!(state, [0], pauli_x(T))
        apply!(state, [0], S; adjoint=true)
        prepare!(state)
        @test collect(only(compute!(state).tensors)) ≈ [0, -im]
    end

    @testset "diagonal operators" begin
        T = ComplexF64
        state = CuNetworkState{T}([2, 2])
        apply!(state, [0], hadamard(T))
        apply!(state, [1], pauli_x(T))
        # CZ flips the sign of |11⟩
        apply_diagonal!(state, [0, 1], CuArray(T[1 1; 1 -1]))
        acc = CuStateAccessor(state)
        prepare!(acc)
        amplitudes, _ = compute!(acc)
        @test collect(amplitudes) ≈ [0 1; 0 -1] ./ sqrt(2)

        # the library doesn't support computing the full state
        prepare!(state)
        @test_broken begin
            compute!(state)
            true
        end
    end

    @testset "controlled operators" begin
        T = ComplexF64
        state = CuNetworkState{T}([2, 2])
        apply!(state, [0], hadamard(T))
        apply_controlled!(state, [0], [1], pauli_x(T); unitary=true)
        prepare!(state)
        @test collect(only(compute!(state).tensors)) ≈ BELL

        # acting when the control is |0⟩ instead
        state = CuNetworkState{T}([2, 2])
        apply_controlled!(state, [0], [1], pauli_x(T); control_values=[0])
        prepare!(state)
        @test collect(only(compute!(state).tensors)) ≈ [0 1; 0 0]
    end

    @testset "host data" begin
        T = ComplexF64
        state = CuNetworkState{T}([2])
        apply!(state, [0], T[0 1; 1 0])
        prepare!(state)
        @test collect(only(compute!(state).tensors)) ≈ [0, 1]
    end
end

@testset "state properties" begin
    @testset for T in [ComplexF32, ComplexF64]
        @testset "amplitudes" begin
            state = bell_state(T)
            acc = CuStateAccessor(state)
            @test_throws ArgumentError compute!(acc)
            prepare!(acc)
            @test flops(acc) >= 0
            amplitudes, norm = compute!(acc)
            @test collect(amplitudes) ≈ BELL
            @test norm ≈ 1

            # with the first mode projected to |1⟩
            acc = CuStateAccessor(state, [0])
            configure!(acc, cuTensorNet.CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES, Int32(1))
            prepare!(acc)
            amplitudes, _ = compute!(acc, [1])
            @test collect(amplitudes) ≈ [0, 1] ./ sqrt(2)
        end

        @testset "expectation values" begin
            state = bell_state(T)
            operator = CuNetworkOperator{T}([2, 2])
            # 2 Z₀Z₁ + 3 X₀
            @test append_product!(operator, 2, [[0], [1]], [pauli_z(T), pauli_z(T)]) == 0
            @test append_product!(operator, 3, [[0]], [pauli_x(T)]) == 1
            @test_throws DimensionMismatch append_product!(operator, 1, [[0]], CuArray{T,2}[])
            x = CuStateExpectation(state, operator)
            configure!(x, cuTensorNet.CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES, Int32(1))
            @test_throws ArgumentError compute!(x)
            prepare!(x)
            @test flops(x) >= 0
            value, norm = compute!(x)
            @test value ≈ 2
            @test norm ≈ 1
        end

        @testset "matrix product operators" begin
            state = bell_state(T)
            operator = CuNetworkOperator{T}([2, 2])
            # Z₀Z₁ with a bond of dimension 1: modes (ket, bond, bra) and (bond, ket, bra)
            Z = T[1 0; 0 -1]
            @test append_mpo!(operator, 1, [0, 1], [CuArray(reshape(Z, 2, 1, 2)), CuArray(reshape(Z, 1, 2, 2))]) == 0
            @test_throws DimensionMismatch append_mpo!(operator, 1, [0], [CuArray(reshape(Z, 2, 1, 2)), CuArray(reshape(Z, 1, 2, 2))])
            x = CuStateExpectation(state, operator)
            prepare!(x)
            value, _ = compute!(x)
            @test value ≈ 1
        end

        @testset "marginals" begin
            state = bell_state(T)
            marginal = CuStateMarginal(state, [0])
            configure!(marginal, cuTensorNet.CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES, Int32(1))
            @test_throws ArgumentError compute!(marginal)
            prepare!(marginal)
            @test flops(marginal) >= 0
            @test marginal_kind(marginal) == cuTensorNet.CUTENSORNET_MARGINAL_KIND_FULL
            @test collect(compute!(marginal)) ≈ [1 0; 0 1] ./ 2

            marginal = CuStateMarginal(state, [0]; diagonal=true)
            prepare!(marginal)
            @test marginal_kind(marginal) == cuTensorNet.CUTENSORNET_MARGINAL_KIND_DIAGONAL
            @test collect(compute!(marginal)) ≈ [1, 1] ./ 2

            # projecting the other mode on |1⟩
            marginal = CuStateMarginal(state, [0]; projected_modes=[1], diagonal=true)
            prepare!(marginal)
            @test collect(compute!(marginal, [1])) ≈ [0, 1] ./ 2
        end

        @testset "sampling" begin
            state = bell_state(T)
            sampler = CuStateSampler(state, [0, 1])
            configure!(sampler, cuTensorNet.CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES, Int32(1))
            configure!(sampler, cuTensorNet.CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC, Int32(42))
            @test_throws ArgumentError sample!(sampler, 1)
            prepare!(sampler)
            @test flops(sampler) >= 0
            samples = sample!(sampler, 100)
            @test size(samples) == (2, 100)
            # the qubits are perfectly correlated, and both outcomes occur
            @test samples[1, :] == samples[2, :]
            @test all(in((0, 1)), samples)
            @test 0 < count(==(1), samples[1, :]) < 100
        end
    end

    @testset "gradients" begin
        T = ComplexF64
        # ⟨ψ|Z|ψ⟩ with |ψ⟩ = G|0⟩ is G[1,1]² - G[2,1]² for real G
        G = CuArray(T[0.6 0.8; 0.8 -0.6])
        state = CuNetworkState{T}([2])
        gradient = CUDACore.zeros(T, 2, 2)
        id = apply_with_gradient!(state, [0], G, gradient; immutable=false)
        operator = CuNetworkOperator{T}([2])
        append_product!(operator, 1, [[0]], [pauli_z(T)])
        x = CuStateExpectation(state, operator)
        prepare!(x)
        value, norm = compute_with_gradients!(x)
        @test value ≈ 0.6^2 - 0.8^2
        @test norm ≈ 1
        forward = copy(collect(gradient))
        @test !iszero(forward)
        @test all(isfinite, forward)

        # accumulating
        compute_with_gradients!(x; accumulate=true)
        @test collect(gradient) ≈ 2 .* forward

        # redirecting the output
        other = CUDACore.zeros(T, 2, 2)
        update_gradient!(state, id, other)
        @test state.gradients[id] === other
        compute_with_gradients!(x)
        @test collect(other) ≈ forward

        # the norm has a different gradient, and the adjoints scale the contributions
        compute_with_gradients!(x; expectation_adjoint=0, norm_adjoint=1)
        norm_gradient = copy(collect(other))
        @test !(norm_gradient ≈ forward)
        compute_with_gradients!(x; expectation_adjoint=2, norm_adjoint=3)
        @test collect(other) ≈ 2 .* forward .+ 3 .* norm_gradient

        # the expectation value can also be computed without them
        @test first(compute!(x)) ≈ 0.6^2 - 0.8^2
    end
end

@testset "channels" begin
    T = ComplexF64
    # bit flip with probability 1: always X
    state = CuNetworkState{T}([2])
    apply_unitary_channel!(state, [0], [pauli_x(T), CuArray(T[1 0; 0 1])], [1.0, 0.0])
    @test_throws DimensionMismatch apply_unitary_channel!(state, [0], [pauli_x(T)], [0.5, 0.5])
    marginal = CuStateMarginal(state, [0]; diagonal=true)
    prepare!(marginal)
    @test collect(compute!(marginal)) ≈ [0, 1]

    # a general channel only works with MPS: the identity Kraus operator leaves the state as is
    state = CuNetworkState{T}([2, 2])
    apply!(state, [0], pauli_x(T))
    apply_general_channel!(state, [0], [CuArray(T[1 0; 0 1])])
    finalize_mps!(state, [[2, 2], [2, 2]])
    prepare!(state)
    compute!(state)
    marginal = CuStateMarginal(state, [0]; diagonal=true)
    prepare!(marginal)
    @test collect(compute!(marginal)) ≈ [0, 1]
end

@testset "MPS" begin
    @testset for T in [ComplexF32, ComplexF64]
        ghz_extents = [[2, 2], [2, 2, 2], [2, 2]]

        function ghz()
            state = CuNetworkState{T}([2, 2, 2])
            apply!(state, [0], hadamard(T))
            apply!(state, [0, 1], cnot(T))
            apply!(state, [1, 2], cnot(T))
            return state
        end
        contract_mps(A, B, C, ext) = [sum(A[i, a] * B[a, j, b] * C[b, k] for a in 1:ext[1][2], b in 1:ext[3][1]) for i in 1:2, j in 1:2, k in 1:2]
        function expected_ghz()
            out = zeros(ComplexF64, 2, 2, 2)
            out[1, 1, 1] = out[2, 2, 2] = 1 / sqrt(2)
            return out
        end

        @testset "factorizing" begin
            state = ghz()
            @test_throws DimensionMismatch finalize_mps!(state, [[2, 2]])
            finalize_mps!(state, ghz_extents)
            configure!(state, cuTensorNet.CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF, 1e-5)
            prepare!(state)
            output = compute!(state)
            @test length(output.tensors) == 3
            A, B, C = collect.(output.tensors)
            @test contract_mps(A, B, C, output.extents) ≈ expected_ghz()
            d = details(state)
            @test length(d.extents) == 3
            @test d.extents == output.extents

            # properties use the factorized state
            marginal = CuStateMarginal(state, [2]; diagonal=true)
            prepare!(marginal)
            @test collect(compute!(marginal)) ≈ [1, 1] ./ 2

            # continue from the computed MPS
            capture_mps!(state)
            @test isempty(state.tensors)
            apply!(state, [2], pauli_x(T))
            prepare!(state)
            A, B, C = collect.(compute!(state).tensors)
            out = zeros(ComplexF64, 2, 2, 2)
            out[1, 1, 2] = out[2, 2, 1] = 1 / sqrt(2)
            @test contract_mps(A, B, C, output.extents) ≈ out
        end

        @testset "initial state" begin
            # |1⟩ ⊗ |0⟩
            state = CuNetworkState{T}([2, 2])
            @test_throws DimensionMismatch initialize_mps!(state, [CUDACore.zeros(T, 2, 1)])
            initialize_mps!(state, [CuArray(reshape(T[0, 1], 2, 1)), CuArray(reshape(T[1, 0], 1, 2))])
            prepare!(state)
            @test collect(only(compute!(state).tensors)) ≈ [0 0; 1 0]
        end
    end
end

@testset "MPS projection" begin
    T = ComplexF64
    # project |00⟩ on an MPS of bond dimension 1, whose dual tensors start out as |00⟩ too
    state = CuNetworkState{T}([2, 2])
    duals() = [CuArray(reshape(T[1, 0], 2, 1)), CuArray(reshape(T[1, 0], 1, 2))]
    env = -1 => 1
    projection = CuStateProjectionMPS(state, duals(), [env])
    configure!(projection, cuTensorNet.CUTENSORNET_STATE_PROJECTION_MPS_CONFIG_NUM_HYPER_SAMPLES, Int32(1))
    @test_throws ArgumentError compute_env!(projection, env, nothing, CUDACore.zeros(T, 2, 1))
    prepare!(projection)

    info = tensor_info(projection, env)
    @test info.extents == [2, 1]
    @test info.strides == [1, 2]

    # environments are computed in a cycle of extracting and inserting the site tensors
    site = CUDACore.zeros(T, info.extents...)
    out = CUDACore.zeros(T, info.extents...)
    extract_tensor!(site, projection, env)
    @test collect(site) ≈ reshape(T[1, 0], 2, 1)
    compute_env!(projection, env, nothing, out)
    @test collect(out) ≈ reshape(T[1, 0], 2, 1)

    # scaling the state scales the environment
    update_coefficient!(projection, 2)
    compute_env!(projection, env, nothing, out)
    @test collect(out) ≈ reshape(T[2, 0], 2, 1)
    update_coefficient!(projection, 1)
    insert_tensor!(projection, env, site)

    # inserted tensors make it to the MPS
    extract_tensor!(site, projection, env)
    insert_tensor!(projection, env, CuArray(reshape(T[0, 1], 2, 1)))
    extract_tensor!(site, projection, env)
    @test collect(site) ≈ reshape(T[0, 1], 2, 1)
    insert_tensor!(projection, env, site)

    # replacing all dual tensors restores the original
    new_duals = duals()
    update_dual_tensors!(projection, new_duals)
    @test projection.data === new_duals
    extract_tensor!(site, projection, env)
    @test collect(site) ≈ reshape(T[1, 0], 2, 1)

    @test_throws ArgumentError CuStateProjectionMPS(state, duals(), [-1 => 2])

    # interior and multi-site environments, with bonds of dimension 2
    state3 = CuNetworkState{T}([2, 2, 2])
    duals3() = [CuArray(T[1 0; 0 1]),
                CuArray(reshape(T[i == j == k ? 1 : 0 for i in 1:2, j in 1:2, k in 1:2], 2, 2, 2)),
                CuArray(T[1 0; 0 1])]
    envs = [-1 => 1, 0 => 2, 1 => 3, -1 => 2, 0 => 1]
    projection = CuStateProjectionMPS(state3, duals3(), envs; ortho=(0 => 2))
    prepare!(projection)
    @test tensor_info(projection, 0 => 2).extents == [2, 2, 2]
    @test tensor_info(projection, -1 => 2).extents == [2, 2, 2]
    @test tensor_info(projection, 0 => 1).extents == [2, 2]
    @test length(tensor_info(projection, 1 => 3).extents) == 2
    # the orthogonality center must end up on the site itself, on one of the two sites, or
    # next to the bond for one-site, two-site and zero-site environments respectively
    for (env, ortho) in ((0 => 2, 0 => 2), (-1 => 2, -1 => 1), (0 => 1, 0 => 2))
        extents = tensor_info(projection, env).extents
        tensor = CUDACore.zeros(T, extents...)
        environment = CUDACore.zeros(T, extents...)
        extract_tensor!(tensor, projection, env)
        compute_env!(projection, env, nothing, environment)
        # the overlap of the MPS and |000⟩ is only made up of the first element
        @test collect(environment)[1] ≈ 1
        @test count(!iszero, round.(collect(environment); digits=8)) == 1
        insert_tensor!(projection, env, tensor; ortho)
    end

    # a symmetric projection takes the initial state from the MPS, and accepts a replacement
    projection = CuStateProjectionMPS(state, duals(), [env]; symmetric=true)
    prepare!(projection)
    extract_tensor!(site, projection, env)
    compute_env!(projection, env, site, out)
    @test collect(out) ≈ reshape(T[1, 0], 2, 1)
end

@testset "finalizing in any order" begin
    # dependent objects can be finalized in the same GC cycle as their state
    for _ in 1:3
        let state = bell_state(ComplexF64)
            prepare!(CuStateAccessor(state))
            prepare!(CuStateSampler(state, [0]))
            operator = CuNetworkOperator{ComplexF64}([2, 2])
            append_product!(operator, 1, [[0]], [pauli_z(ComplexF64)])
            prepare!(CuStateExpectation(state, operator))
            prepare!(CuStateMarginal(state, [0]))
            prepare!(CuStateProjectionMPS(state, [CuArray(reshape(ComplexF64[1, 0], 2, 1)), CuArray(reshape(ComplexF64[1, 0], 1, 2))], [-1 => 1]))

            other = CuNetworkState{ComplexF64}([2, 2])
            apply!(other, operator)
        end
        GC.gc(true)
        # (some of the destructors only run when reclaiming)
        CUDACore.reclaim()
    end
    @test true
end
