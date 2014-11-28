if length(opts["server"]) > 0
    using JSON
    using HTTPClient.HTTPC

    # Setup codespeed data dict for submissions to codespeed's JSON endpoint.  These parameters
    # are constant across all benchmarks, so we'll just let them sit here for now
    pkgdir = dirname(Base.source_path())
    csdata = Dict()
    csdata["commitid"] = chomp(readall(`git -C $pkgdir rev-parse HEAD`))
    csdata["project"] = "CUDA.jl"
    csdata["branch"] = chomp(readall(`git -C $pkgdir symbolic-ref -q --short HEAD`))
    csdata["executable"] = CUDA_VENDOR
    csdata["environment"] = chomp(readall(`hostname`))
    csdata["result_date"] = join( split(chomp(readall(`git -C $pkgdir log --pretty=format:%cd -n 1 --date=iso`)))[1:2], " " )    #Cut the timezone out
end

# Takes in the raw array of values in vals, along with the benchmark name, description, unit and whether less is better
function submit_to_codespeed(vals,name,desc,unit,test_group,lessisbetter=true)
    for host in opts["server"]
        csdata["benchmark"] = name
        csdata["description"] = desc
        csdata["result_value"] = mean(vals)
        csdata["std_dev"] = std(vals)
        csdata["min"] = minimum(vals)
        csdata["max"] = maximum(vals)
        csdata["units"] = unit
        csdata["units_title"] = test_group
        csdata["lessisbetter"] = lessisbetter

        println( "$name: $(mean(vals))" )
        ret = post( "http://$host/result/add/json/", Dict("json" => json([csdata])) )
        println( json([csdata]) )
        if ret.http_code != 200 && ret.http_code != 202
            error("Error submitting $name [HTTP code $(ret.http_code)], dumping headers and text: $(ret.headers)\n$(bytestring(ret.body))\n\n")
        end
    end
end

function readable(d)
    if d > 60
        error("unimplemented")
    elseif d < 1
        t = ["m", "µ", "n", "p"]
        for i in 1:length(t)
            scale = 10.0^-3i
            if scale < d <= scale*1000
                return "$(signif(d/scale, 2)) $(t[i])s"
            end
        end
        error()
    else
        return "$(round(d, 2)) s"
    end
end

macro output_timings(t,name,desc,group)
    quote
        # If we weren't given anything for the test group, infer off of file path!
        test_group = length($group) == 0 ? basename(dirname(Base.source_path())) : $group[1]
        if length(opts["server"]) > 0
            submit_to_codespeed($t, $name, $desc, "seconds", test_group)
        else
            @printf "%-20s: %s ± %s\n" $name readable(mean($t)) readable(std($t))
        end
        gc()
    end
end

const mintrials = 5
const maxtime = 2.5       # in seconds

macro timeit(ex,name,desc,group...)
    quote
        @timeit_init $ex begin end $name $desc $group...
    end
end

macro timeit_init(ex,init,name,desc,group...)
    quote
        trials = mintrials
        t = zeros(trials)
        start = time()
        i = 0
        while i <= trials
            gc_disable()
            $(esc(init))
            e = @elapsed $(esc(ex))
            gc_enable()
            if i > 0
                # warm up on first iteration
                t[i] = e
            end
            if i == trials && (time()-start) < maxtime
                # check if accurate enough
                uncertainty = std(t[1:i])/mean(t[1:i])
                if uncertainty > .05
                    trials *= 2
                    resize!(t, trials)
                end
            end

            i += 1
        end
        @output_timings t $name $desc $group
    end
end


# seed rng for more consistent timings
srand(1776)
