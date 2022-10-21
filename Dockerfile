# example of a Docker container for CUDA.jl with a specific toolkit embedded at run time.

FROM julia:1.8-bullseye


# system-wide packages

ENV JULIA_DEPOT_PATH=/usr/local/share/julia

RUN julia -e 'using Pkg; Pkg.add("CUDA")'

# hard-code a CUDA toolkit version
RUN julia -e 'using CUDA; CUDA.set_runtime_version!(v"11.8")'
# re-importing CUDA.jl below will trigger a download of the relevant artifacts

# generate the device runtime library for all known and supported devices.
# this is to avoid having to do this over and over at run time.
RUN julia -e 'using CUDA; CUDA.precompile_runtime()' && \
    chmod 644 /usr/local/share/julia/compiled/v1.8/GPUCompiler/*/*.bc
    # TODO: fix this in GPUCompiler.jl


# user environment

# we hard-code the primary depot regardless of the actual user, i.e., we do not let it
# default to `$HOME/.julia`. this is for compatibility with `docker run --user`, in which
# case there might not be a (writable) home directory.

RUN mkdir -m 0777 /depot
ENV JULIA_DEPOT_PATH=/depot:/usr/local/share/julia

WORKDIR "/workspace"
