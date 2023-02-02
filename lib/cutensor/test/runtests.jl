using Test

# work around JuliaLang/Pkg.jl#2500
if VERSION < v"1.8"
    test_project = first(Base.load_path())
    preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
    test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
    if isfile(preferences_file) && !isfile(test_preferences_file)
        cp(preferences_file, test_preferences_file)
    end
end

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
