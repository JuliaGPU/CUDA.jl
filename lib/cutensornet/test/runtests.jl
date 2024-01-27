using Test

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using LinearAlgebra
using cuTENSOR
using cuTensorNet
@test cuTENSOR.has_cutensor()
@test cuTensorNet.has_cutensornet()
@info "cuTensorNet version: $(cuTensorNet.version()) (built for CUDA $(cuTensorNet.cuda_version()))"

import cuTensorNet: CuTensorNetwork, rehearse_contraction, perform_contraction!, gateSplit!, AutoTune, NoAutoTune, amplitudes!, CuState, applyTensor!, cutensornetTensorQualifiers_t, SamplerConfig, AccessorConfig, MarginalConfig, ExpectationConfig, expectation, CuNetworkOperator, appendToOperator!, compute_marginal!, sample!

using TensorOperations

@testset "cuTensorNet" begin
    n = 8
    m = 16
    k = 32
    #=@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "Simple serial" begin
            modesA = ['m', 'h', 'k', 'n']
            modesB = ['u', 'k', 'h']
            modesC = ['x', 'u', 'y']
            modesD = ['m', 'x', 'n', 'y']
            extent = Dict{Char, Int}()
            extent['m'] = 96;
            extent['n'] = 96;
            extent['u'] = 96;
            extent['h'] = 64;
            extent['k'] = 64;
            extent['x'] = 64;
            extent['y'] = 64;
            extentsA = [extent[mode] for mode in modesA]
            extentsB = [extent[mode] for mode in modesB]
            extentsC = [extent[mode] for mode in modesC]
            extentsD = [extent[mode] for mode in modesD]
            A = CUDA.rand(elty, extentsA...)
            B = CUDA.rand(elty, extentsB...)
            C = CUDA.rand(elty, extentsC...)
            raw_data_in = [A, B, C]
            modes_in = [Int32.(modesA), Int32.(modesB), Int32.(modesC)]
            extents_in = [extentsA, extentsB, extentsC]
            aligns_in = UInt32.([256, 256, 256])
            aligns_out = UInt32(256)
            strides_in = [Int32.(collect(strides(A))), Int32.(collect(strides(B))), Int32.(collect(strides(C)))]
            qualifiers_in = [cutensornetTensorQualifiers_t(0, 0, 0), cutensornetTensorQualifiers_t(0, 0, 0), cutensornetTensorQualifiers_t(0, 0, 0)] 
            ctn = CuTensorNetwork(elty, modes_in, extents_in, strides_in, qualifiers_in, Int32.(modesD), extentsD, C_NULL)
            @testset for max_ws_size in [2^28, 2^32]
                @testset for tuning in [NoAutoTune(), AutoTune()]
                    ctn.input_arrs = raw_data_in
                    info = rehearse_contraction(ctn, max_ws_size)
                    ctn.output_arr = CUDA.zeros(elty, extentsD...)
                    ctn = perform_contraction!(ctn, info, tuning)
                    @test size(ctn.output_arr) == tuple(extentsD...)
                    hA = collect(A)
                    hB = collect(B)
                    hC = collect(C)
                    hD = zeros(elty, extentsD...)
                    @tensor begin
                        hD[m, x, n, y] := hA[m, h, k, n] * hB[u, k, h] * hC[x, u, y]
                    end
                    D = collect(ctn.output_arr)
                    @test D ≈ hD
                end
            end
        end
        @testset "QR" begin
            A = CUDA.rand(elty, n, m)
            modesA = ['n','m']
            Q = CUDA.zeros(elty, n, n)
            R = CUDA.zeros(elty, n, m)
            Q, R = qr!(CuTensor(A, modesA), CuTensor(Q, ['n', 'o']), CuTensor(R, ['o', 'm']))
            @test collect(Q*R) ≈ collect(A)
        end
        @testset "SVD" begin
            A = CUDA.rand(elty, n, n)
            modesA = ['n','m']
            U = CUDA.zeros(elty, n, n)
            S = CUDA.zeros(real(elty), n)
            V = CUDA.zeros(elty, n, n)
            config = cuTensorNet.SVDConfig(abs_cutoff=0.0, rel_cutoff=0.0)
            U, S, V, info = svd!(CuTensor(A, modesA), CuTensor(U, ['n', 'o']), S, CuTensor(V, ['o', 'm']), svd_config=config)
            @test cuTensorNet.full_extent(info)      == n
            @test cuTensorNet.reduced_extent(info)   == n
            @test cuTensorNet.discarded_weight(info) ≈ 0.0
            @test collect(U)*diagm(collect(S))*collect(V) ≈ collect(A)
        end
        @testset "GateSplit" begin
            a = 16
            b = 16
            c = 16
            d = 2
            f = 2
            i = 2
            j = 2
            g = 16
            h = 16
            A = CUDA.rand(elty, a, b, c, d)
            B = CUDA.rand(elty, c, f, g, h)
            G = CUDA.rand(elty, i, j, d, f)
            modesA = ['a','b','c','d']
            modesB = ['c','f','g','h']
            modesG = ['i','j','d','f']
            z = 16
            Aout = CUDA.zeros(elty, a, b, z, i)
            modesAout = ['a','b','z','i']
            S    = CUDA.zeros(real(elty), z)
            Bout = CUDA.zeros(elty, z, j, g, h)
            modesBout = ['z','j','g','h']
            config = cuTensorNet.SVDConfig(abs_cutoff=0.0, rel_cutoff=0.0)
            Aout, S, Bout, info = gateSplit!(CuTensor(A, modesA), CuTensor(B, modesB), CuTensor(G, modesG), CuTensor(Aout, modesAout), S, CuTensor(Bout, modesBout), svd_config=config)
            @test cuTensorNet.full_extent(info)      == z*z*2
            @test cuTensorNet.reduced_extent(info)   == z
        end
    end=#
    @testset "Amplitudes" begin
        h_gate = 1/√2 * ComplexF64[1. 1.; 1. -1.]
        c_not_gate = ComplexF64[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
        n_amps = 1
        n_qubits = 6
        h_tensor  = CuTensor(CuMatrix{ComplexF64}(h_gate), [Char(0)])
        cx_tensors = [CuTensor(CuMatrix{ComplexF64}(c_not_gate), Char[i, i+1]) for i in 0:n_qubits-2]
        qubit_dims = [(i, 2) for i in 0:n_qubits-1]
        state = CuState{ComplexF64}(qubit_dims)
        fixed_modes = [0, 1, 2, 3, 4, 5]
        fixed_mode_vals = Int64[1, 1, 1, 1, 1, 1]
        applyTensor!(state, h_tensor, true)
        for cxt in cx_tensors
            applyTensor!(state, cxt, true)
        end
        amp_tensor = CuTensor(CUDA.zeros(ComplexF64, 2^(n_qubits - length(fixed_modes))), Char.(sort(setdiff(0:n_qubits-1, fixed_modes))))
        GC.@preserve state amp_tensor begin
            amp_tensor, norm = amplitudes!(state, fixed_modes, fixed_mode_vals, amp_tensor; config=AccessorConfig(num_hyper_samples=8))
            @test norm == 1.0
            @test collect(amp_tensor.data) ≈ [1.0/√2]
        end
    end
    @testset "Sampling" begin
        h_gate = 1/√2 * ComplexF64[1. 1.; 1. -1.]
        c_not_gate = ComplexF64[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
        n_shots    = 100
        n_qubits   = 16
        h_tensor   = CuTensor(CuMatrix{ComplexF64}(h_gate), [Char(0)])
        cx_tensors = [CuTensor(CuMatrix{ComplexF64}(c_not_gate), Char[i, i+1]) for i in 0:n_qubits-2]
        qubit_dims = [(i, 2) for i in 0:n_qubits-1]
        state      = CuState{ComplexF64}(qubit_dims)
        applyTensor!(state, h_tensor, true)
        for cxt in cx_tensors
            applyTensor!(state, cxt, true)
        end
        samples = sample!(state, collect(0:n_qubits-1), n_shots; config=SamplerConfig(num_hyper_samples=8))
        @test all(s ∈ [ones(Int, n_qubits), zeros(Int, n_qubits)] for s in eachcol(samples))
    end
    @testset "Marginal" begin
        h_gate = 1/√2 * ComplexF64[1. 1.; 1. -1.]
        c_not_gate = ComplexF64[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
        n_qubits = 6
        h_tensor  = CuTensor(CuMatrix{ComplexF64}(h_gate), [Char(0)])
        cx_tensors = [CuTensor(CuMatrix{ComplexF64}(c_not_gate), Char[i, i+1]) for i in 0:n_qubits-2]
        qubit_dims = [(i, 2) for i in 0:n_qubits-1]
        state = CuState{ComplexF64}(qubit_dims)
        marginal_modes = [0, 1]
        applyTensor!(state, h_tensor, true)
        for cxt in cx_tensors
            applyTensor!(state, cxt, true)
        end
        marginal_tensor = CuTensor(CUDA.zeros(ComplexF64, 2^(length(marginal_modes)) * 2^(length(marginal_modes))), Char.(marginal_modes))
        GC.@preserve state marginal_tensor begin
            marginal_tensor = compute_marginal!(state, marginal_modes, Int[], Int[], marginal_tensor; config=MarginalConfig(num_hyper_samples=8))
        end
        h_marginal_tensor = collect(marginal_tensor.data)
        correct_tensor = zeros(Float64, 2^(n_qubits - length(marginal_modes)))
        correct_tensor[1] = 0.5
        correct_tensor[end] = 0.5
        @test h_marginal_tensor ≈ correct_tensor
    end
    @testset "Expectation" begin
        h_gate     = 1/√2 * ComplexF64[1. 1.; 1. -1.]
        c_not_gate = ComplexF64[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
        n_qubits   = 4
        h_tensor   = CuTensor(CuMatrix{ComplexF64}(h_gate), [Char(0)])
        cx_tensors = [CuTensor(CuMatrix{ComplexF64}(c_not_gate), Char[i, i+1]) for i in 0:n_qubits-2]
        qubit_dims = [(i, 2) for i in 0:n_qubits-1]
        state = CuState{ComplexF64}(qubit_dims)
        applyTensor!(state, h_tensor, true)
        for cxt in cx_tensors
            applyTensor!(state, cxt, true)
        end
        network_op = CuNetworkOperator{ComplexF64}(qubit_dims)
        z_gate = ComplexF64[1 0; 0 -1]
        y_gate = ComplexF64[0 -im; im 0]
        x_gate = ComplexF64[0 1; 1 0]
        i_gate = ComplexF64[1 0; 0 1]
        z1_tensor  = CuTensor(CuMatrix{ComplexF64}(z_gate), [Char(1)])
        z2_tensor  = CuTensor(CuMatrix{ComplexF64}(z_gate), [Char(2)])
        y3_tensor  = CuTensor(CuMatrix{ComplexF64}(y_gate), [Char(3)])

        y0_tensor  = CuTensor(CuMatrix{ComplexF64}(y_gate), [Char(0)])
        x2_tensor  = CuTensor(CuMatrix{ComplexF64}(x_gate), [Char(2)])
        z3_tensor  = CuTensor(CuMatrix{ComplexF64}(z_gate), [Char(3)])
        
        i0_tensor  = CuTensor(CuMatrix{ComplexF64}(i_gate), [Char(0)])
        i1_tensor  = CuTensor(CuMatrix{ComplexF64}(i_gate), [Char(1)])
        i2_tensor  = CuTensor(CuMatrix{ComplexF64}(i_gate), [Char(2)])
        i3_tensor  = CuTensor(CuMatrix{ComplexF64}(i_gate), [Char(3)])
        #network_op = appendToOperator!(network_op, ComplexF64(0.5), [z1_tensor, z2_tensor])
        #network_op = appendToOperator!(network_op, ComplexF64(0.25), [y3_tensor])
        #network_op = appendToOperator!(network_op, ComplexF64(0.13), [y0_tensor, x2_tensor, z3_tensor])
        network_op = appendToOperator!(network_op, ComplexF64(1.0), [i0_tensor, i1_tensor, i2_tensor, i3_tensor])
        GC.@preserve state network_op begin
            exp, norm = expectation(state, network_op; config=ExpectationConfig(num_hyper_samples=8))
            @test exp ≈ 1.0 
            @test norm ≈ 1.0 
        end 
    end
end
