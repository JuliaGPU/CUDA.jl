#!/usr/bin/env julia

# Script to parse and compare the libdevice PDF manual against our list of intrinsics

function parse_intrinsics(cb)
    fn = joinpath(@__DIR__, "..", "src", "device", "libdevice.jl")
    open(fn) do f
        for ln in eachline(f)
            m = match(r"@wrap ([\w.]+\(.+?\)::\w+)", ln)
            if m != nothing
                cb(replace(m.captures[1], r"\w+::", "::"))
            end
        end
    end
end

function parse_libdevice(fn, cb)
    open(fn) do f
        next_proto = false
        number = 0

        for ln in eachline(f)
            if (m = match(r"^\d\.(\d+)\..", ln); m != nothing)
                number = parse(Int, m.captures[1])
            elseif ismatch(r"^Prototype:", ln)
                next_proto = true
            elseif next_proto
                cb(chomp(ln), number)
                next_proto = false
            end
        end
    end
end

function main(args)
    if length(args) != 1
        println("Usage: $(basename(@__FILE__)) LIBDEVICE_PDF")
        exit(1)
    end
    pdf = args[1]
    isfile(pdf) || error("input PDF does not exist")

    wrapped = Set{String}()
    parse_intrinsics(intr -> push!(wrapped, intr))

    intrinsics = Set{String}()
    numbering = Dict{String,Number}()
    txt = tempname()
    run(`pdftotext $pdf $txt`)
    parse_libdevice(txt, (proto, number) -> begin
        m = match(r"^(\w+) (@[\w.]+)\((.*?)\)", proto)
        if m != nothing
            rettype = m.captures[1]
            fn = m.captures[2]
            arglist = m.captures[3]

            argpairs = split(arglist, ", ")
            argtypes, args = zip(map(argpair -> split(argpair, " "), argpairs)...)

            wrap_fn = strip(fn, '@')
            wrap_argtypes = map(argtyp -> endswith(argtyp, '*') ? "Ptr{$(argtyp[1:end-1])}"
                                                                : argtyp, argtypes)
            wrap_args = map(arg -> strip(arg, '%'), args)
            wrap_arglist = join(["$arg::$argtyp" for (arg, argtyp) in zip(wrap_args, wrap_argtypes)], ", ")

            intr = "$wrap_fn($wrap_arglist)::$rettype"
            push!(intrinsics, intr)
            numbering[intr] = number
        end
    end)
    rm(txt)

    missing = setdiff(intrinsics, wrapped)
    superfluous = setdiff(wrapped, intrinsics)

    println("Missing intrinsics:")
    for intr in sort(collect(missing), lt=(a,b)->numbering[a]<numbering[b])
        println(" $(numbering[intr]). $intr")
    end

    println()

    println("Superfluous intrinsics:")
    for intr in sort(collect(superfluous))
        println(" - $intr")
    end
end

main(ARGS)
