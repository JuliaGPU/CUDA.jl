#!/usr/bin/env julia

# Release script for CUDA.jl monorepo.
#
# Updates all Project.toml version fields and internal compat bounds to the given
# version, then prints `@JuliaRegistrator register` commands in dependency order
# (with "wait" markers between waves).
#
# Usage:
#   julia res/release.jl 6.1.0

using TOML

if length(ARGS) != 1
    println(stderr, "Usage: julia res/release.jl <version>")
    exit(1)
end

const new_version = ARGS[1]

# validate version string
if !occursin(r"^\d+\.\d+\.\d+$", new_version)
    println(stderr, "Error: version must be in X.Y.Z format, got: $new_version")
    exit(1)
end

const root = dirname(@__DIR__)

# All packages in the monorepo: (name, subdir relative to root, Project.toml path)
const packages = [
    ("CUDACore",    "CUDACore",          joinpath(root, "CUDACore", "Project.toml")),
    ("CUDATools",   "CUDATools",         joinpath(root, "CUDATools", "Project.toml")),
    ("CUPTI",       "lib/cupti",         joinpath(root, "lib", "cupti", "Project.toml")),
    ("NVML",        "lib/nvml",          joinpath(root, "lib", "nvml", "Project.toml")),
    ("cuBLAS",      "lib/cublas",        joinpath(root, "lib", "cublas", "Project.toml")),
    ("cuSPARSE",    "lib/cusparse",      joinpath(root, "lib", "cusparse", "Project.toml")),
    ("cuFFT",       "lib/cufft",         joinpath(root, "lib", "cufft", "Project.toml")),
    ("cuRAND",      "lib/curand",        joinpath(root, "lib", "curand", "Project.toml")),
    ("cuDNN",       "lib/cudnn",         joinpath(root, "lib", "cudnn", "Project.toml")),
    ("cuTENSOR",    "lib/cutensor",      joinpath(root, "lib", "cutensor", "Project.toml")),
    ("cuStateVec",  "lib/custatevec",    joinpath(root, "lib", "custatevec", "Project.toml")),
    ("cuSOLVER",    "lib/cusolver",      joinpath(root, "lib", "cusolver", "Project.toml")),
    ("cuTensorNet", "lib/cutensornet",   joinpath(root, "lib", "cutensornet", "Project.toml")),
    ("CUDA",        "",                  joinpath(root, "Project.toml")),
]

# The set of all internal package names (for identifying internal compat entries)
const internal_names = Set(first.(packages))

# --- Step 1: Patch all Project.toml files ---

"""
    patch_toml(text, new_version, internal_names) -> (text, changed)

Update the version field and internal compat bounds in a Project.toml string.
Only modifies entries under `[compat]` to avoid touching UUIDs in `[deps]`.
"""
function patch_toml(text::String, new_version::String, internal_names::Set{String})
    lines = split(text, '\n')
    changed = false
    in_compat = false

    for (i, line) in enumerate(lines)
        # Track which TOML section we're in
        m = match(r"^\[(\w+)\]", line)
        if m !== nothing
            in_compat = m[1] == "compat"
            continue
        end

        # Update top-level version = "..."
        vm = match(r"^version\s*=\s*\"([^\"]+)\"", line)
        if vm !== nothing && vm[1] != new_version
            lines[i] = "version = \"$new_version\""
            changed = true
            continue
        end

        # Update compat bounds for internal deps
        if in_compat
            cm = match(r"^(\w+)\s*=\s*\"([^\"]+)\"", line)
            if cm !== nothing && cm[1] in internal_names
                new_compat = "=$new_version"
                if cm[2] != new_compat
                    lines[i] = "$(cm[1]) = \"=$new_version\""
                    changed = true
                end
            end
        end
    end

    return join(lines, '\n'), changed
end

for (name, subdir, toml_path) in packages
    text = read(toml_path, String)
    text, changed = patch_toml(text, new_version, internal_names)

    if changed
        write(toml_path, text)
        println("Updated $toml_path")
    else
        println("No changes needed for $toml_path")
    end
end

# --- Step 2: Print registration commands in dependency order ---
#
# Wave 1: CUDACore (no internal deps)
# Wave 2: CUPTI, NVML, cuBLAS, cuSPARSE, cuFFT, cuRAND, cuDNN, cuTENSOR, cuStateVec
#          (depend only on CUDACore)
# Wave 3: CUDATools (depends on CUPTI, NVML),
#          cuSOLVER (depends on cuBLAS, cuSPARSE),
#          cuTensorNet (depends on cuTENSOR)
# Wave 4: CUDA (depends on everything above)

const waves = [
    ["CUDACore"],
    ["CUPTI", "NVML", "cuBLAS", "cuSPARSE", "cuFFT", "cuRAND", "cuDNN", "cuTENSOR", "cuStateVec"],
    ["CUDATools", "cuSOLVER", "cuTensorNet"],
    ["CUDA"],
]

# Build a lookup from name to subdir
const subdir_map = Dict(name => subdir for (name, subdir, _) in packages)

println()
println("=" ^ 60)
println("Registration commands (post as GitHub comments in order)")
println("=" ^ 60)

for (i, wave) in enumerate(waves)
    if i > 1
        println()
        println("--- wait for wave $(i-1) to be merged in the registry ---")
        println()
    end
    for name in wave
        sd = subdir_map[name]
        if isempty(sd)
            println("@JuliaRegistrator register")
        else
            println("@JuliaRegistrator register subdir=\"$sd\"")
        end
    end
end
