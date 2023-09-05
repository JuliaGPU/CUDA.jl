using Test

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using cuTENSOR
@test cuTENSOR.has_cutensor()
@info "cuTENSOR version: $(cuTENSOR.version()) (built for CUDA $(cuTENSOR.cuda_version()))"

@testset "cuTENSOR" begin

# include all tests
for entry in readdir(@__DIR__)
    endswith(entry, ".jl") || continue
    entry in ["runtests.jl"] && continue

    # generate a testset
    name = splitext(entry)[1]
    @eval begin
        @testset $name begin
            include($entry)
        end
    end
end

end
