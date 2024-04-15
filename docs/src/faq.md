# Frequently Asked Questions

This page is a compilation of frequently asked questions and answers.


## An old version of CUDA.jl keeps getting installed!

Sometimes it happens that a breaking version of CUDA.jl or one of its dependencies is
released. If any package you use isn't yet compatible with this release, this will block
automatic upgrade of CUDA.jl. For example, with Flux.jl v0.11.1 we get CUDA.jl v1.3.3
despite there being a v2.x release:

```
pkg> add Flux
  [587475ba] + Flux v0.11.1
pkg> add CUDA
  [052768ef] + CUDA v1.3.3
```

To examine which package is holding back CUDA.jl, you can "force" an upgrade by specifically
requesting a newer version. The resolver will then complain, and explain why this upgrade
isn't possible:

```
pkg> add CUDA.jl@2
  Resolving package versions...
ERROR: Unsatisfiable requirements detected for package Adapt [79e6a3ab]:
 Adapt [79e6a3ab] log:
 ├─possible versions are: [0.3.0-0.3.1, 0.4.0-0.4.2, 1.0.0-1.0.1, 1.1.0, 2.0.0-2.0.2, 2.1.0, 2.2.0, 2.3.0] or uninstalled
 ├─restricted by compatibility requirements with CUDA [052768ef] to versions: [2.2.0, 2.3.0]
 │ └─CUDA [052768ef] log:
 │   ├─possible versions are: [0.1.0, 1.0.0-1.0.2, 1.1.0, 1.2.0-1.2.1, 1.3.0-1.3.3, 2.0.0-2.0.2] or uninstalled
 │   └─restricted to versions 2 by an explicit requirement, leaving only versions 2.0.0-2.0.2
 └─restricted by compatibility requirements with Flux [587475ba] to versions: [0.3.0-0.3.1, 0.4.0-0.4.2, 1.0.0-1.0.1, 1.1.0] — no versions left
   └─Flux [587475ba] log:
     ├─possible versions are: [0.4.1, 0.5.0-0.5.4, 0.6.0-0.6.10, 0.7.0-0.7.3, 0.8.0-0.8.3, 0.9.0, 0.10.0-0.10.4, 0.11.0-0.11.1] or uninstalled
     ├─restricted to versions * by an explicit requirement, leaving only versions [0.4.1, 0.5.0-0.5.4, 0.6.0-0.6.10, 0.7.0-0.7.3, 0.8.0-0.8.3, 0.9.0, 0.10.0-0.10.4, 0.11.0-0.11.1]
     └─restricted by compatibility requirements with CUDA [052768ef] to versions: [0.4.1, 0.5.0-0.5.4, 0.6.0-0.6.10, 0.7.0-0.7.3, 0.8.0-0.8.3, 0.9.0, 0.10.0-0.10.4] or uninstalled, leaving only versions: [0.4.1, 0.5.0-0.5.4, 0.6.0-0.6.10, 0.7.0-0.7.3, 0.8.0-0.8.3, 0.9.0, 0.10.0-0.10.4]
       └─CUDA [052768ef] log: see above
```

A common source of these incompatibilities is having both CUDA.jl and the older
CUDAnative.jl/CuArrays.jl/CUDAdrv.jl stack installed: These are incompatible, and cannot
coexist. You can inspect in the Pkg REPL which exact packages you have installed using the
`status --manifest` option.


## Can you wrap this or that CUDA API?

If a certain API isn't wrapped with some high-level functionality, you can always use the
underlying C APIs which are always available as unexported methods. For example, you can
access the CUDA driver library as `cu` prefixed, unexported functions like
`CUDA.cuDriverGetVersion`. Similarly, vendor libraries like CUBLAS are available through
their exported submodule handles, e.g., `CUBLAS.cublasGetVersion_v2`.

Any help on designing or implementing high-level wrappers for this low-level functionality
is greatly appreciated, so please consider contributing your uses of these APIs on the
respective repositories.


## When installing CUDA.jl on a cluster, why does Julia stall during precompilation?

If you're working on a cluster, precompilation may stall if you have not requested 
sufficient memory. You may also wish to make sure you have enough disk space prior
to installing CUDA.jl.
