using LinearAlgebra: mul!
using StaticArrays

@testset "StaticArrays" begin
    function batched_matvec(ms::CuArray, vs::CuArray)
        function matvec_kernel(out, ms, vs, ::Val{N}, ::Val{M}) where {N, M}
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            # Call constructors without @inbounds.
            # This asserts that the @device_override
            # for StaticArrays.dimension_mismatch_fail() works.
            m = SMatrix{N, M, Float32}(@view ms[:, :, i])
            v = SVector{M, Float32}(@view vs[:, i])
            out[:, i] .= m * v
            nothing
        end

        out = similar(ms, (size(ms, 1), size(ms, 3)))
        @cuda threads=size(ms, 3) matvec_kernel(out, ms, vs, Val(size(ms, 1)), Val(size(ms, 2)))
        out
    end

    function batched_matvec(ms, vs)
        out = similar(ms, (size(ms, 1), size(ms, 3)))
        foreach((o, m, v) -> mul!(o, m, v), eachcol(out), eachslice(ms; dims=3), eachcol(vs))
        out
    end

    ms, vs = randn(Float32, 3, 2, 4), randn(Float32, 2, 4)
    @test batched_matvec(ms, vs) ≈ Array(batched_matvec(cu(ms), cu(vs)))
end