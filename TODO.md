* Merge with pointers in CUDArt.jl

* Define `hash` methods for all types with `==` defined (see
  [JuliaLang/julia#12198](https://github.com/JuliaLang/julia/issues/12198)

* Verify that objects with implicit links to the current context actually keep the context
  alive (eg. `CuModule`, `CuFunction`)

* Put clean-up calls in finalizers (eg. `CuStream`, `CuLinker`, ...)

* Some `CuModule` ctors can return objects with identical handles -- are the deduplicated by
  the driver, or are we doing something wrong? In the case of the former, can we safely free?
