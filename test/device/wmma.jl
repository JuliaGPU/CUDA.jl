using CUDA.WMMA

################################################################################

@testset "LLVM intrinsics" begin
    @testset "llvm_wmma_load" begin
        @testset "$(mat)_$(layout)_$(shape)_$(addr_space)_$(elem_type)" for mat in ["a", "b", "c"],
            layout in ["row", "col"],
            shape in ["m16n16k16"],
            addr_space in ["", "_global", "_shared"],
            stride in ["stride"],
            elem_type in ["f16", "f32"]

            # Float32 is only supported for C
            if (elem_type == "f32") && (mat != "c")
                continue
            end

            # Type-dependent variables
            array_ty = elem_type == "f16" ? Float16 : Float32
            expected = elem_type == "f16" ? ntuple(i -> VecElement{Float16}(42), 2) : Float32(42)

            # Address-space dependent variables
            do_shared_test = (addr_space == "_shared")

            # Get the function name
            func = Symbol("llvm_wmma_load_$(mat)_$(layout)_$(shape)$(addr_space)_stride_$(elem_type)")

            input      = 42 * ones(array_ty, (16, 16))
            input_dev  = CuArray(input)
            result     = Array{Bool}(undef, 1)
            result_dev = CuArray(result)

            @eval @inbounds function kernel(input_ptr, result_dev)
                if $do_shared_test
                    input_shared = @cuStaticSharedMem($array_ty, 256)
                    fill!(input_shared, 42)

                    data = $func(pointer(input_shared), 16)
                else
                    data = $func(input_ptr, 16)
                end

                result_dev[1] = all(val -> val == $expected, data)

                return
            end

            # FIXME: make the intrinsics dispatch on the address-space
            #        (but how to test AS.Generic then, since CuArray adapts to a global ptr?)
            input_ptr = if addr_space == ""
                reinterpret(Core.LLVMPtr{array_ty,AS.Generic}, pointer(input_dev))
            elseif addr_space == "_global"
                reinterpret(Core.LLVMPtr{array_ty,AS.Global}, pointer(input_dev))
            else
                nothing
            end

            @cuda threads=32 kernel(input_ptr, result_dev)
            @test all(Array(result_dev))
        end
    end

    @testset "llvm_wmma_store" begin
        @testset "$(mat)_$(layout)_$(shape)_$(addr_space)_$(elem_type)" for mat in ["d"],
            layout in ["row", "col"],
            shape in ["m16n16k16"],
            addr_space in ["", "_global", "_shared"],
            stride in ["stride"],
            elem_type in ["f16", "f32"]

            # Type-dependent variables
            array_ty = elem_type == "f16" ? Float16 : Float32
            data = elem_type == "f16" ? ntuple(i -> ntuple(j -> VecElement{Float16}(42), 2), 4) : ntuple(i -> 42, 8)

            # Get the function name
            func = Symbol("llvm_wmma_store_$(mat)_$(layout)_$(shape)$(addr_space)_stride_$(elem_type)")

            # Address-space dependent variables
            do_shared_test = (addr_space == "_shared")

            output     = Array{array_ty}(undef, (16, 16))
            output_dev = CuArray(output)

            @eval function kernel(output_dev, output_ptr)
                if $do_shared_test
                    shared_mem = @cuStaticSharedMem($array_ty, 256)
                    $func(pointer(shared_mem), $data, 16)

                    for i = 1:256
                        @inbounds output_dev[i] = shared_mem[i]
                    end
                else
                    $func(output_ptr, $data, 16)
                end

                return
            end

            # FIXME: make the intrinsics dispatch on the address-space
            #        (but how to test AS.Generic then, since CuArray adapts to a global ptr?)
            output_ptr = if addr_space == ""
                reinterpret(Core.LLVMPtr{array_ty,AS.Generic}, pointer(output_dev))
            elseif addr_space == "_global"
                reinterpret(Core.LLVMPtr{array_ty,AS.Global}, pointer(output_dev))
            else
                nothing
            end

            @cuda threads=32 kernel(output_dev, output_ptr)
            @test all(Array(output_dev) .== 42.0)
        end
    end

    @testset "llvm_wmma_mma" begin
        @testset "$(a_layout)_$(b_layout)_$(shape)_$(d_elem_type)_$(c_elem_type)" for a_layout in ["row", "col"],
            b_layout in ["row", "col"],
            shape in ["m16n16k16"],
            d_elem_type in ["f16", "f32"],
            c_elem_type in ["f16", "f32"]

            # Type-dependent variables
            d_ty = d_elem_type == "f16" ? Float16 : Float32
            c_ty = c_elem_type == "f16" ? Float16 : Float32

            # Get the function names
            lda_func = getfield(Main, Symbol("llvm_wmma_load_a_$(a_layout)_m16n16k16_global_stride_f16"))
            ldb_func = getfield(Main, Symbol("llvm_wmma_load_b_$(b_layout)_m16n16k16_global_stride_f16"))
            ldc_func = getfield(Main, Symbol("llvm_wmma_load_c_col_m16n16k16_global_stride_$(c_elem_type)"))
            mma_func = getfield(Main, Symbol("llvm_wmma_mma_$(a_layout)_$(b_layout)_m16n16k16_$(d_elem_type)_$(c_elem_type)"))
            std_func = getfield(Main, Symbol("llvm_wmma_store_d_col_m16n16k16_global_stride_$(d_elem_type)"))

            # Generate input matrices
            a     = rand(Float16, (16, 16))
            a_dev = CuArray(a)
            b     = rand(Float16, (16, 16))
            b_dev = CuArray(b)
            c     = rand(c_ty, (16, 16))
            c_dev = CuArray(c)

            # Reserve space for result
            d     = Array{d_ty}(undef, (16, 16))
            d_dev = CuArray(d)

            # Matrix MAC kernel (D = A * B + C)
            function kernel(a_dev, b_dev, c_dev, d_dev)
                a_frag = lda_func(pointer(a_dev), 16)
                b_frag = ldb_func(pointer(b_dev), 16)
                c_frag = ldc_func(pointer(c_dev), 16)

                d_frag = mma_func(a_frag, b_frag, c_frag)

                std_func(pointer(d_dev), d_frag, 16)
                return
            end

            @cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)

            new_a = (a_layout == "col" ? a : transpose(a))
            new_b = (b_layout == "col" ? b : transpose(b))

            @test all(isapprox.(new_a * new_b + c, Array(d_dev); rtol=sqrt(eps(Float16))))
        end
    end
end

################################################################################

@testset "Flattening/unflattening" begin
    @testset "Flattening" begin
        @test WMMA.flatten(5)                                                                  == (5,)
        @test WMMA.flatten(5.0)                                                                == (5.0,)
        @test WMMA.flatten(VecElement{Float16}(5))                                             == (Float16(5),)
        @test WMMA.flatten(ntuple(i -> i, 8))                                                  == ntuple(i -> i, 8)
        @test WMMA.flatten(ntuple(i -> VecElement{Float16}(i), 8))                             == ntuple(i -> Float16(i), 8)
        @test WMMA.flatten(ntuple(i -> ntuple(j -> (i-1) * 2 + j, 2), 8))                      == ntuple(i -> i, 2 * 8)
        @test WMMA.flatten(ntuple(i -> ntuple(j -> VecElement{Float16}((i-1) * 2 + j), 2), 8)) == ntuple(i -> Float16(i), 2 * 8)
    end

    @testset "Unflattening" begin
        @test WMMA.unflatten(Int64, (5,))                                                               == 5
        @test WMMA.unflatten(Float64, (5.0,))                                                           == 5.0
        @test WMMA.unflatten(VecElement{Float16}, (Float16(5),))                                        == VecElement{Float16}(5)
        @test WMMA.unflatten(NTuple{8, Int64}, ntuple(i -> i, 8))                                       == ntuple(i -> i, 8)
        @test WMMA.unflatten(NTuple{8, VecElement{Float16}}, ntuple(i -> Float16(i), 8))                == ntuple(i -> VecElement{Float16}(i), 8)
        @test WMMA.unflatten(NTuple{8, NTuple{2, Int64}}, ntuple(i -> i, 2 * 8))                        == ntuple(i -> ntuple(j -> (i-1) * 2 + j, 2), 8)
        @test WMMA.unflatten(NTuple{8, NTuple{2, VecElement{Float16}}}, ntuple(i -> Float16(i), 2 * 8)) == ntuple(i -> ntuple(j -> VecElement{Float16}((i-1) * 2 + j), 2), 8)
    end
end

################################################################################

@testset "Broadcasting over fragments: size=$sz, type=$ty" for sz = [1, 2, 5],
        ty = [Float16, Float32]
        @test ty(5) .* Fragment{16, 16, 16, sz, ty, RowMajor, MatrixA}(ntuple(i -> ty(i), sz)) == Fragment{16, 16, 16, sz, ty, RowMajor, MatrixA}(ntuple(i -> ty(5 * i), sz))
        @test ty(5) .+ Fragment{16, 16, 16, sz, ty, RowMajor, MatrixA}(ntuple(i -> ty(i), sz)) == Fragment{16, 16, 16, sz, ty, RowMajor, MatrixA}(ntuple(i -> ty(5 + i), sz))
end

################################################################################

@testset "CUDA C-style API" begin
    @testset "$(do_mac ? "MAC" : "MUL"): A: $a_layout, B: $b_layout, C: $c_layout, D: $d_layout, C type: $c_type, D type: $d_type" for a_layout in [ColMajor, RowMajor],
        b_layout in [ColMajor, RowMajor],
        c_layout in [ColMajor, RowMajor],
        d_layout in [ColMajor, RowMajor],
        c_type in [Float16, Float32],
        d_type in [Float16, Float32],
        do_mac in [true, false]

        a     = rand(Float16, (16, 16))
        b     = rand(Float16, (16, 16))
        c     = rand(c_type, (16, 16))
        d     = Array{d_type}(undef, (16, 16))

        a_dev = CuArray(a)
        b_dev = CuArray(b)
        c_dev = CuArray(c)
        d_dev = CuArray(d)

        alpha = rand(Float16)
        beta  = rand(c_type)

        @eval function kernel(a_dev, b_dev, c_dev, d_dev, alpha, beta)
            conf = Config{16, 16, 16, $d_type}

            a_frag = load_a(pointer(a_dev), 16, $a_layout, conf)
            b_frag = load_b(pointer(b_dev), 16, $b_layout, conf)

            if $do_mac
                c_frag = load_c(pointer(c_dev), 16, $c_layout, conf)
            else
                c_frag = fill_c($c_type(0), conf)
            end

            a_frag = alpha .* a_frag
            c_frag = beta .* c_frag

            d_frag = mma(a_frag, b_frag, c_frag, conf)

            store_d(pointer(d_dev), d_frag, 16, $d_layout, conf)

            return
        end

        @cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev, alpha, beta)
        d = Array(d_dev)

        new_a = (a_layout == ColMajor) ? a : transpose(a)
        new_b = (b_layout == ColMajor) ? b : transpose(b)
        new_c = (c_layout == ColMajor) ? c : transpose(c)
        new_d = (d_layout == ColMajor) ? d : transpose(d)

        if do_mac
            @test all(isapprox.(alpha * new_a * new_b + beta * new_c, new_d; rtol=sqrt(eps(Float16))))
        else
            @test all(isapprox.(alpha * new_a * new_b, new_d; rtol=sqrt(eps(Float16))))
        end
    end

end

################################################################################

@testset "Codegen addressing" begin
    @testset "Global" begin
        function kernel(d)
            conf = WMMA.Config{16, 16, 16, Float32}

            d_frag = WMMA.fill_c(Float32(0), conf)
            WMMA.store_d(pointer(d), d_frag, 16, WMMA.ColMajor, conf)

            return
        end

        ptx = sprint(io -> CUDA.code_ptx(io, kernel, (CuDeviceArray{Float32,1,CUDA.AS.Global},)))

        @test !occursin(r"wmma.store.d.sync(.aligned)?.col.m16n16k16.f32", ptx)
        @test  occursin(r"wmma.store.d.sync(.aligned)?.col.m16n16k16.global.f32", ptx)
    end

    @testset "Shared" begin
        function kernel()
            shmem = @cuStaticSharedMem(Float32, (16, 16))
            conf = WMMA.Config{16, 16, 16, Float32}

            d_frag = WMMA.fill_c(Float32(0), conf)
            WMMA.store_d(pointer(shmem), d_frag, 16, WMMA.ColMajor, conf)

            return
        end

        ptx = sprint(io -> CUDA.code_ptx(io, kernel, ()))

        @test !occursin(r"wmma.store.d.sync(.aligned)?.col.m16n16k16.f32", ptx)
        @test  occursin(r"wmma.store.d.sync(.aligned)?.col.m16n16k16.shared.f32", ptx)
    end
end
