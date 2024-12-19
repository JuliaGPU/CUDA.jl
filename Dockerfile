# example of a Docker container for CUDA.jl with a specific toolkit embedded at run time.

ARG JULIA_VERSION=1
FROM julia:${JULIA_VERSION}

ARG CUDA_VERSION=12.6

ARG PACKAGE_SPEC=CUDA

LABEL maintainer="Tim Besard <tim.besard@gmail.com>"
LABEL description="CUDA.jl container with CUDA ${CUDA_VERSION} installed for Julia ${JULIA_VERSION}"
LABEL version="1.0"


# system-wide packages

# no trailing ':' as to ensure we don't touch anything outside this directory. without it,
# Julia touches the compilecache timestamps in its shipped depot (for some reason; a bug?)
ENV JULIA_DEPOT_PATH=/usr/local/share/julia

# pre-install the CUDA toolkit from an artifact. we do this separately from CUDA.jl so that
# this layer can be cached independently. it also avoids double precompilation of CUDA.jl in
# order to call `CUDA.set_runtime_version!`.
RUN julia -e '#= configure the preference =# \
              env = "/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)"; \
              mkpath(env); \
              write("$env/LocalPreferences.toml", \
                    "[CUDA_Runtime_jll]\nversion = \"'${CUDA_VERSION}'\""); \
              \
              #= install the JLL =# \
              using Pkg; \
              Pkg.add("CUDA_Runtime_jll")' && \
    #= remove nondeterminisms =# \
    cd /usr/local/share/julia && \
    rm -rf compiled registries scratchspaces logs && \
    find -exec touch -h -d "@0" {} + && \
    touch -h -d "@0" /usr/local/share

# install CUDA.jl itself
RUN julia -e 'using Pkg; pkg"add '${PACKAGE_SPEC}'"; \
              using CUDA; CUDA.precompile_runtime()' && \
    #= remove useless stuff =# \
    cd /usr/local/share/julia && \
    rm -rf registries scratchspaces logs


# user environment

# we hard-code the primary depot regardless of the actual user, i.e., we do not let it
# default to `$HOME/.julia`. this is for compatibility with `docker run --user`, in which
# case there might not be a (writable) home directory.

RUN mkdir -m 0777 /depot
ENV JULIA_DEPOT_PATH=/depot:/usr/local/share/julia:

# make sure the user depot is the one used by default (which requires a valid Project.toml)
RUN cp -ar /usr/local/share/julia/environments /depot

WORKDIR "/workspace"
