# example of a Docker container for CUDA.jl with a specific toolkit embedded at run time.
#
# supports selecting a Julia and CUDA toolkit version, as well as baking in specific
# versions of the CUDA.jl package (identified by a spec recorgnized by the Pkg REPL).
#
# CUDA.jl and other packages are shipped in a system depot, with the user depot mounted
# at `/depot`. persistency is possible by mounting a volume at this location.
# running with reduced privileges (by using `--user`) is also supported.

ARG JULIA_VERSION=1
FROM julia:${JULIA_VERSION}

ARG JULIA_CPU_TARGET=native
ENV JULIA_CPU_TARGET=${JULIA_CPU_TARGET}

ARG CUDA_VERSION=12.6

ARG PACKAGE_SPEC=CUDA

LABEL org.opencontainers.image.authors="Tim Besard <tim.besard@gmail.com>" \
      org.opencontainers.image.description="A CUDA.jl container with CUDA ${CUDA_VERSION} and Julia ${JULIA_VERSION}" \
      org.opencontainers.image.title="CUDA.jl" \
      org.opencontainers.image.url="https://juliagpu.org/cuda/" \
      org.opencontainers.image.source="https://github.com/JuliaGPU/CUDA.jl" \
      org.opencontainers.image.licenses="MIT"


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
    #= demote the JLL to an [extras] dep =# \
    find /usr/local/share/julia/environments -name Project.toml -exec sed -i 's/deps/extras/' {} + && \
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

# we add the user environment from a start-up script
# so that the user can mount `/depot` for persistency
ENV JULIA_DEPOT_PATH=/usr/local/share/julia:
COPY <<EOF /usr/local/share/julia/config/startup.jl
if !isdir("/depot/environments/v$(VERSION.major).$(VERSION.minor)")
    if isinteractive() && Base.JLOptions().quiet == 0
        println("""Welcome to this CUDA.jl container!

                   Since this is the first time you're running this container,
                   we'll set up a user depot for you at `/depot`. For persistency,
                   you can mount a volume at this location.

                   The CUDA.jl package is pre-installed, and ready to be imported.
                   Remember that you need to invoke Docker with e.g. `--gpus=all`
                   to access the GPU.""")
    end
    mkpath("/depot/environments")
    cp("/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)",
       "/depot/environments/v$(VERSION.major).$(VERSION.minor)")
end
pushfirst!(DEPOT_PATH, "/depot")
EOF

WORKDIR "/workspace"
