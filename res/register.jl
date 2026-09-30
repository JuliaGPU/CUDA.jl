#!/usr/bin/env julia

# Tagging script for the CUDA.jl monorepo.
#
# Posts `@JuliaRegistrator register` comments on a commit, wave by wave in dependency
# order (see res/release.jl), waiting for each registration PR to be created and merged
# in the General registry before proceeding to the next wave.
#
# Usage:
#   res/tag.jl [--skip=Pkg1,Pkg2] <commit> [< release_notes]
#
# Release notes can be passed on stdin and will be attached to every registration:
#   res/tag.jl abc123 <<EOD
#   Breaking changes: ...
#   EOD
#
# The script is resumable: packages whose version is already in the General registry are
# skipped, and register comments that were already posted on the commit are not posted
# again (the existing registration PR is located and waited upon instead). After a failed
# registration, fix the problem and simply re-run the script with the same arguments.
# Use --skip to manually exclude packages on top of that.
#
# Requires the `gh` CLI, authenticated with permission to comment on the repository.

using TOML
using Base64

const repo = "JuliaGPU/CUDA.jl"
const registry = "JuliaRegistries/General"
const registrator_user = "JuliaRegistrator"

const root = dirname(@__DIR__)

# All packages in the monorepo: name => subdir relative to root ("" for the root package)
const subdirs = Dict(
    "CUDACore"    => "CUDACore",
    "CUDATools"   => "CUDATools",
    "CUPTI"       => "lib/cupti",
    "NVML"        => "lib/nvml",
    "cuBLAS"      => "lib/cublas",
    "cuSPARSE"    => "lib/cusparse",
    "cuFFT"       => "lib/cufft",
    "cuRAND"      => "lib/curand",
    "cuDNN"       => "lib/cudnn",
    "cuTENSOR"    => "lib/cutensor",
    "cuStateVec"  => "lib/custatevec",
    "cuSOLVER"    => "lib/cusolver",
    "cuTensorNet" => "lib/cutensornet",
    "CUDA"        => "",
)

# Registration order; packages within a wave only depend on earlier waves
const waves = [
    ["CUDACore"],
    ["CUPTI", "NVML", "cuBLAS", "cuSPARSE", "cuFFT", "cuRAND", "cuDNN", "cuTENSOR", "cuStateVec"],
    ["CUDATools", "cuSOLVER", "cuTensorNet"],
    ["CUDA"],
]


## gh CLI helpers

function gh(args...)
    out = IOBuffer()
    err = IOBuffer()
    proc = run(pipeline(ignorestatus(`gh $(collect(args))`); stdout=out, stderr=err))
    if !success(proc)
        error("gh $(join(args, ' ')) failed:\n$(String(take!(err)))")
    end
    return String(take!(out))
end

# like gh(), but returns nothing on failure (e.g. 404)
function gh_maybe(args...)
    try
        return gh(args...)
    catch
        return nothing
    end
end


## registry queries

# check whether a package version is already registered in General
function registered(name, version)
    path = "$(uppercase(name[1]))/$name/Versions.toml"
    out = gh_maybe("api", "-H", "Accept: application/vnd.github.raw",
                   "repos/$registry/contents/$path")
    out === nothing && return false   # package not registered at all
    return haskey(TOML.parse(out), version)
end

const pr_titles = Dict{Int,String}()
pr_title(pr) = get!(pr_titles, pr) do
    strip(gh("api", "repos/$registry/pulls/$pr", "--jq", ".title"))
end

function wait_for_merge(name, pr)
    print("  $name: waiting for $registry#$pr to be merged ")
    while true
        out = strip(gh("api", "repos/$registry/pulls/$pr",
                       "--jq", "[.state, (.merged|tostring)] | @tsv"))
        state, merged = split(out, '\t')
        if merged == "true"
            println(" merged!")
            return
        elseif state == "closed"
            println()
            error("registration PR https://github.com/$registry/pull/$pr for $name was closed without being merged")
        end
        print(".")
        flush(stdout)
        sleep(60)
    end
end


## commit comments

struct Comment
    id::Int
    user::String
    body::String
end

function get_comments(sha)
    out = gh("api", "repos/$repo/commits/$sha/comments", "--paginate",
             "--jq", ".[] | [.id, .user.login, (.body|@base64)] | @tsv")
    comments = Comment[]
    for line in eachline(IOBuffer(out))
        parts = split(line, '\t')
        length(parts) == 3 || continue
        push!(comments, Comment(parse(Int, parts[1]), parts[2],
                                String(base64decode(parts[3]))))
    end
    return comments
end

register_command(subdir) =
    isempty(subdir) ? "@JuliaRegistrator register" :
                      "@JuliaRegistrator register subdir=\"$subdir\""

# find a comment whose first line is exactly the given register command
# (exact match, since the plain `register` command is a prefix of the subdir ones)
function find_register_comment(comments, cmd)
    for c in comments
        first_line = strip(first(split(c.body, '\n')))
        first_line == cmd && return c
    end
    return nothing
end

# find the registrator reply announcing the registration PR for a given package,
# by matching the PR title (e.g. "New version: CUDACore v6.2.0") against the package
function find_registration_pr(comments, name, version)
    for c in comments
        c.user == registrator_user || continue
        m = match(r"JuliaRegistries/General/pull/(\d+)", c.body)
        m === nothing && continue
        pr = parse(Int, m[1])
        if occursin(Regex("^New (version|package): \\Q$name\\E v\\Q$version\\E\$"), pr_title(pr))
            return pr
        end
    end
    return nothing
end

# registrator comments that do not contain a PR link are (presumably) errors
registrator_errors(comments) =
    filter(comments) do c
        c.user == registrator_user &&
        match(r"JuliaRegistries/General/pull/(\d+)", c.body) === nothing
    end


## main

function usage()
    println(stderr, "Usage: res/tag.jl [--skip=Pkg1,Pkg2] <commit> [< release_notes]")
    exit(1)
end

function main(args)
    skipped = String[]
    commit = nothing
    for arg in args
        if startswith(arg, "--skip=")
            append!(skipped, split(chopprefix(arg, "--skip="), ','))
        elseif startswith(arg, "-")
            usage()
        elseif commit === nothing
            commit = arg
        else
            usage()
        end
    end
    commit === nothing && usage()
    for name in skipped
        haskey(subdirs, name) || error("unknown package in --skip: $name")
    end

    # release notes from stdin, if redirected
    notes = if stdin isa Base.TTY
        nothing
    else
        str = strip(read(stdin, String))
        isempty(str) ? nothing : String(str)
    end

    # resolve the commit and read each package's version from it
    sha = strip(gh("api", "repos/$repo/commits/$commit", "--jq", ".sha"))
    println("Tagging $repo@$(sha[1:10])")
    notes !== nothing && println("Release notes will be attached to every registration.")
    versions = Dict{String,String}()
    for (name, subdir) in subdirs
        path = isempty(subdir) ? "Project.toml" : "$subdir/Project.toml"
        project = TOML.parse(readchomp(`git -C $root show $sha:$path`))
        project["name"] == name ||
            error("$path at $sha contains package $(project["name"]), expected $name")
        versions[name] = project["version"]
    end
    println()

    for (i, wave) in enumerate(waves)
        println("Wave $i: $(join(wave, ", "))")

        # figure out which packages still need to be registered
        pending = String[]
        for name in wave
            if name in skipped
                println("  $name: skipped")
            elseif registered(name, versions[name])
                println("  $name: v$(versions[name]) already registered")
            else
                push!(pending, name)
            end
        end
        if isempty(pending)
            println()
            continue
        end

        # post register comments for packages that don't have one yet
        comments = get_comments(sha)
        known_errors = Set(c.id for c in registrator_errors(comments))
        for name in pending
            cmd = register_command(subdirs[name])
            if find_register_comment(comments, cmd) !== nothing
                println("  $name: register comment already posted")
                continue
            end
            body = cmd
            if notes !== nothing
                body *= "\n\nRelease notes:\n\n" * notes
            end
            gh("api", "repos/$repo/commits/$sha/comments", "-f", "body=$body",
               "--jq", ".id")
            println("  $name: posted register comment")
        end

        # wait for the registrator to reply with registration PRs
        prs = Dict{String,Int}()
        print("  waiting for JuliaRegistrator ")
        while length(prs) < length(pending)
            print(".")
            flush(stdout)
            sleep(15)

            comments = get_comments(sha)
            for name in pending
                haskey(prs, name) && continue
                pr = find_registration_pr(comments, name, versions[name])
                if pr !== nothing
                    prs[name] = pr
                    print(" $name: $registry#$pr ")
                end
            end

            # any new registrator comment without a PR link indicates failure
            errors = filter(c -> c.id ∉ known_errors, registrator_errors(comments))
            if !isempty(errors)
                println()
                for c in errors
                    println(stderr, "\nJuliaRegistrator reported an error:\n")
                    println(stderr, c.body)
                end
                error("registration failed; fix the problem and re-run to resume")
            end
        end
        println()

        # wait for all registration PRs to be merged
        for name in pending
            wait_for_merge(name, prs[name])
        end
        println()
    end

    println("All done! TagBot will create the tags and GitHub releases shortly.")
end

main(ARGS)
