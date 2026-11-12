#!/usr/bin/env julia

# Release script for CUDA.jl monorepo.
#
# Updates all Project.toml version fields and internal compat bounds to the given
# version. Once the result is merged, register the packages with res/register.jl.
#
# Usage:
#   julia res/bump_version.jl 6.1.0

using TOML

if length(ARGS) != 1
    println(stderr, "Usage: julia res/bump_version.jl <version>")
    exit(1)
end

const new_version = ARGS[1]

# validate version string
if !occursin(r"^\d+\.\d+\.\d+$", new_version)
    println(stderr, "Error: version must be in X.Y.Z format, got: $new_version")
    exit(1)
end

const root = dirname(@__DIR__)

include("workspace.jl")
const packages = workspace_packages() do subdir
    path = joinpath(root, subdir, "Project.toml")
    isfile(path) ? TOML.parsefile(path) : nothing
end

# The set of all internal package names (for identifying internal compat entries)
const internal_names = Set(p.name for p in packages)

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

for (; subdir) in packages
    toml_path = joinpath(root, subdir, "Project.toml")
    text = read(toml_path, String)
    text, changed = patch_toml(text, new_version, internal_names)

    if changed
        write(toml_path, text)
        println("Updated $toml_path")
    else
        println("No changes needed for $toml_path")
    end
end
