using CUDA, CUDACore, SIMD, BFloat16s, Test

@testset "math lowering policy" begin
    llvm = CUDACore.LLVM
    CUDACore.GPUCompiler.JuliaContext() do ctx
        for (flags, fastmath, symbol) in (("", false, "__nv_expf"),
                                         ("afn ", false, "__nv_fast_expf"),
                                         ("", true, "__nv_fast_expf"))
            mod = parse(llvm.Module, """
                declare float @llvm.exp.f32(float)
                define float @entry(float %value) {
                    %result = call $(flags)float @llvm.exp.f32(float %value)
                    ret float %result
                }
                """)
            try
                job = (config=(target=(cap=v"7.5", ptx=v"7.0", fastmath=fastmath),),)
                CUDACore.lower_math_intrinsics!(job, mod)
                @test occursin("call float @$symbol", string(mod))
                @test llvm.verify(mod) === nothing
            finally
                llvm.dispose(mod)
            end
        end
        if tryparse(llvm.Intrinsic, "llvm.tanh") !== nothing
            for ptx in (v"6.3", v"6.5", v"7.0"), cap in (v"7.0", v"7.5")
                mod = parse(llvm.Module, """
                    declare float @llvm.tanh.f32(float)
                    define float @entry(float %value) {
                        %result = call afn float @llvm.tanh.f32(float %value)
                        ret float %result
                    }
                    """)
                try
                    job = (config=(target=(; cap, ptx, fastmath=false),),)
                    CUDACore.lower_math_intrinsics!(job, mod)
                    native = cap >= v"7.5" && ptx >= v"7.0"
                    @test occursin("call afn float @llvm.tanh", string(mod)) == native
                    @test occursin("call float @__nv_tanhf", string(mod)) != native
                finally
                    llvm.dispose(mod)
                end
            end
        end
    end
end

@generated function raw_math(value::Vec{width,typ}, ::Val{operation},
                             exponent::Int32) where {width,typ,operation}
    element = typ === Float16 ? "half" : typ === Float32 ? "float" : "double"
    suffix = typ === Float16 ? "f16" : typ === Float32 ? "f32" : "f64"
    vector = "<$width x $element>"
    name = "llvm.$operation.v$width$suffix"
    extra_type, extra_arg = if operation === :powi
        name *= ".i32"
        (", i32", ", i32 %exponent")
    elseif operation in (:pow, :atan2)
        (", $vector", ", $vector %value")
    else
        ("", "")
    end
    ir = if operation === :sincos
        """
        declare {$vector, $vector} @$name($vector)
        define $vector @entry($vector %value, i32 %exponent) {
            %pair = call {$vector, $vector} @$name($vector %value)
            %sine = extractvalue {$vector, $vector} %pair, 0
            %cosine = extractvalue {$vector, $vector} %pair, 1
            %result = fadd $vector %sine, %cosine
            ret $vector %result
        }
        """
    else
        """
        declare $vector @$name($vector$extra_type)
        define $vector @entry($vector %value, i32 %exponent) {
            %result = call $vector @$name($vector %value$extra_arg)
            ret $vector %result
        }
        """
    end
    quote
        Vec(Base.llvmcall(($ir, "entry"), NTuple{$width,VecElement{$typ}},
                         Tuple{NTuple{$width,VecElement{$typ}},Int32}, value.data, exponent))
    end
end

function raw_math_kernel!(output, input, operation, exponent, ::Val{width}) where {width}
    value = Vec(ntuple(lane -> @inbounds(input[lane]), Val(width)))
    result = raw_math(value, operation, exponent)
    for lane in 1:width
        @inbounds output[lane] = result[lane]
    end
    return
end

function powi_reference_kernel!(output, input)
    for index in eachindex(input)
        value = input[index]
        promoted = value isa Float16 ? Float32(value) : value
        output[index] = CUDACore.powi(promoted, Int32(3))
    end
    return
end

@testset "math intrinsic coverage" begin
    for operation in values(CUDACore.MATH_INTRINSICS)
        tryparse(CUDACore.LLVM.Intrinsic, "llvm.$operation") === nothing && continue
        for (typ, width) in ((Float32, 4), (Float64, 2), (Float16, 4))
            input = CuArray(typ.(range(0.25, 0.75; length=width)))
            output = similar(input)
            @cuda threads=1 raw_math_kernel!(output, input, Val(operation), Int32(3), Val(width))
            reference = if operation === :pow
                Array(input .^ input)
            elseif operation === :powi
                scalar_output = similar(input)
                @cuda threads=1 powi_reference_kernel!(scalar_output, input)
                Array(scalar_output)
            elseif operation === :atan2
                Array(atan.(input, input))
            elseif operation === :sincos
                Array((value -> +(sincos(value)...)).(input))
            else
                Array(getfield(Base, operation).(input))
            end
            @test isequal(Array(output), reference)
        end
    end
end

function vector_math_kernel!(output, input, operation, ::Val{width}) where {width}
    value = Vec(ntuple(lane -> @inbounds(input[lane]), Val(width)))
    result = operation(value)
    for lane in 1:width
        @inbounds output[lane] = result[lane]
    end
    return
end

@testset "LLVM vector math" begin
    for (typ, width) in ((Float32, 4), (Float64, 2), (Float16, 4))
        input = CuArray(typ.(range(0.25, 0.75; length=width)))
        output = similar(input)
        for operation in (exp, exp2, log, log2, log10, sin, cos)
            @cuda threads=1 vector_math_kernel!(output, input, operation, Val(width))
            reference = Array(operation.(input))
            @test isequal(Array(output), reference)
        end
    end
end

@testset "exhaustive half math" begin
    for typ in (Float16, BFloat16)
        typ === BFloat16 && !BFloat16s.llvm_arithmetic && continue
        values = collect(reinterpret(typ, collect(UInt16(0):typemax(UInt16))))
        input = CuArray(values)
        for operation in (exp, exp2, exp10, log, log2, log10, sin, cos)
            output = Array(operation.(input))
            reference = map(values) do value
                if typ === BFloat16 && operation === exp10 &&
                   reinterpret(UInt16, value) == 0xbc95
                    return reinterpret(BFloat16, 0x3f75)
                end
                promoted = BigFloat(value)
                operation in (log, log2, log10) && promoted < 0 && return typ(NaN)
                operation in (sin, cos) && isinf(promoted) && return typ(NaN)
                typ(operation(promoted))
            end
            @testset "$typ $operation" begin
                mismatches = findall(eachindex(output)) do index
                    actual, expected = output[index], reference[index]
                    !((isnan(actual) && isnan(expected)) || isequal(actual, expected))
                end
                if !isempty(mismatches)
                    indices = first(mismatches, min(10, length(mismatches)))
                    @info "half math mismatches" typ operation count=length(mismatches) inputs=values[indices] actual=output[indices] expected=reference[indices]
                end
                @test isempty(mismatches)
            end
        end
    end
end
