@testset "reflection" begin
    dummy() = return

    CUDA.code_lowered(dummy, Tuple{})
    CUDA.code_typed(dummy, Tuple{})
    CUDA.code_warntype(devnull, dummy, Tuple{})
    CUDA.code_llvm(devnull, dummy, Tuple{})
    CUDA.code_ptx(devnull, dummy, Tuple{})
    if can_use_cupti() && !(v"2024.2.0" <= CUPTI.library_version() < v"2025.1.0") # NVIDIA bug #4667039
        # functions defined in Julia
        sass = sprint(io->CUDA.code_sass(io, dummy, Tuple{}))
        @test occursin(".text._Z5dummy", sass)

        # external functions
        sass = sprint(io->begin
            CUDA.code_sass(io) do
                cuBLAS.copy!(1, CUDA.ones(1), CUDA.ones(1))
            end
        end)
        @test occursin("copy_kernel", sass)
    end

    @device_code_lowered @cuda dummy()
    @device_code_typed @cuda dummy()
    @device_code_warntype io=devnull @cuda dummy()
    @device_code_llvm io=devnull @cuda dummy()
    @device_code_ptx io=devnull @cuda dummy()
    if can_use_cupti() && !(v"2024.2.0" <= CUPTI.library_version() < v"2025.1.0") # NVIDIA bug #4667039
        # functions defined in Julia
        sass = sprint(io->@device_code_sass io=io @cuda dummy())
        @test occursin(".text._Z5dummy", sass)

        # external functions
        sass = sprint(io->begin
            @device_code_sass io=io begin
                cuBLAS.copy!(1, CUDA.ones(1), CUDA.ones(1))
            end
        end)
        @test occursin("copy_kernel", sass)

        # inlined frames are shown as a tree, like `@device_code_llvm debuginfo=:source`.
        # this needs line tables with inlining information (-g1, CUDA 11.2+).
        if Base.JLOptions().debug_level == 1 && CUDA.compiler_version() >= v"11.2"
            helper_line = @__LINE__() + 1
            @inline reflection_helper(x) = x * x
            function reflection_kernel(a)
                @inbounds a[1] = reflection_helper(a[1])
                return
            end
            sass = sprint(io->CUDA.code_sass(io, reflection_kernel,
                                             Tuple{CuDeviceVector{Float32,1}}))
            @test occursin(Regex("; ┌ @ .*reflection\\.jl:$helper_line\\n"), sass)
        end
    end

    mktempdir() do dir
        @device_code dir=dir @cuda dummy()
    end

    @test_throws ErrorException @device_code_lowered nothing

    # make sure kernel name aliases are preserved in the generated code
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io optimize=false @cuda dummy())))
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io @cuda dummy())))
    @test occursin("dummy", sprint(io->(@device_code_ptx io=io @cuda dummy())))
    if can_use_cupti() && !(v"2024.2.0" <= CUPTI.library_version() < v"2025.1.0") # NVIDIA bug #4667039
        @test occursin("dummy", sprint(io->(@device_code_sass io=io @cuda dummy())))
    end

    # make sure invalid kernels can be partially reflected upon
    let
        invalid_kernel() = throw()
        @test_throws CUDA.InvalidIRError @cuda invalid_kernel()
        @test_throws CUDA.InvalidIRError @grab_output @device_code_warntype @cuda invalid_kernel()
        out, err = @grab_output begin
            try
                @device_code_warntype @cuda invalid_kernel()
            catch
            end
        end
        @test occursin("Body::Union{}", err)
    end

    # set name of kernel
    @test occursin("mykernel", sprint(io->(@device_code_llvm io=io begin
        k = cufunction(dummy, name="mykernel")
        k()
    end)))

    @test CUDA.return_type(identity, Tuple{Int}) === Int
    @test CUDA.return_type(sin, Tuple{Float32}) === Float32
    @test CUDA.return_type(getindex, Tuple{CuDeviceArray{Float32,1,1},Int32}) === Float32
    @test CUDA.return_type(getindex, Tuple{Base.RefValue{Integer}}) === Integer
end
