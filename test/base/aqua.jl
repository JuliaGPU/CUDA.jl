using Aqua

# FIXME: Adapt.WrappedArray contains subtypes that do not bind the N typevar
#Aqua.test_unbound_args(CUDA)

# FIXME: we have plenty of ambiguities, let's at least ensure that we don't create more
#Aqua.test_ambiguities(CUDA)
let ambs = Aqua.detect_ambiguities(CUDA; recursive=true)
    pkg_match(pkgname, pkgdir::Nothing) = false
    pkg_match(pkgname, pkgdir::AbstractString) = occursin(pkgname, pkgdir)
    filter!(x -> pkg_match("CUDA", pkgdir(last(x).module)), ambs)
    @test length(ambs) ≤ 51
end

Aqua.test_undefined_exports(CUDA)
Aqua.test_stale_deps(CUDA; ignore=[:CUDA_Runtime_Discovery, :CUDA_Runtime_jll,
                                   :SpecialFunctions])
Aqua.test_deps_compat(CUDA)
Aqua.test_project_extras(CUDA)
Aqua.test_project_toml_formatting(CUDA)
Aqua.test_piracy(CUDA)
