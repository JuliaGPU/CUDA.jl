using Aqua

# FIXME: we have plenty of ambiguities, let's at least ensure that we don't create more
let ambs = Aqua.detect_ambiguities(CUDA; recursive=true)
    pkg_match(pkgname, pkgdir::Nothing) = false
    pkg_match(pkgname, pkgdir::AbstractString) = occursin(pkgname, pkgdir)

    # StaticArrays pirates a bunch of Random stuff...
    filter!(x -> !pkg_match("StaticArrays", pkgdir(first(x).module)), ambs)

    @test length(ambs) ≤ 15
end

Aqua.test_all(CUDA;
    stale_deps=(ignore=[:CUDA_Runtime_Discovery, :CUDA_Runtime_jll,
                        :SpecialFunctions],),

    # tested above
    ambiguities=false,

    # FIXME: Adapt.WrappedArray contains subtypes that do not bind the N typevar
    #Aqua.test_unbound_args(CUDA)
    unbound_args=false
)
