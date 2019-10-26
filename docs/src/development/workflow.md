# Recommended workflow

## Package versions

The packages that make up the Julia/CUDA stack are often developed in tandem, and as a
result the master branch of a package might depend on an unreleased version of its
dependencies. Generally, it is recommended to just check-out, or develop the different
packages. Alternatively, most packages provide a top-level Manifest.toml that records a
known-good state of the package and its dependencies.
