using Test
using CUDA
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    # Aqua.test_unbound_args(CUDA)
    ua = Aqua.detect_unbound_args_recursively(CUDA)
    @info "Number of unbound argument methods: $(length(ua))"
    @test length(ua) ≤ 26

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(CUDA; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("CUDA", pkgdir(last(x).module)), ambs)

    # Uncomment for debugging:
    # for method_ambiguity in ambs
    #     @show method_ambiguity
    # end
    @test length(ambs) ≤ 35
    @info "Number of method ambiguities: $(length(ambs))"
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(CUDA)
    # Aqua.test_stale_deps(CUDA) # failing
    Aqua.test_deps_compat(CUDA)
    Aqua.test_project_extras(CUDA)
    Aqua.test_project_toml_formatting(CUDA)
    # Aqua.test_piracy(CUDA) # failing
end

nothing
