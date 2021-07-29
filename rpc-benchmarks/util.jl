
MEASURES = [(:median, Statistics.median), (:mean, Statistics.mean), (:std, Statistics.std), (:max, (args) -> max(args...)), (:min, (args) -> min(args...))]

trial_header(base, measures) = join(["$(base)_$(x)" for (x, _) in MEASURES if x in measures || :all in measures],  ",")

function trial_to_string(trial, measures)
    return join([string(to_msecs(f(skipmissing(trial.times)))) for (x, f) in MEASURES if x in measures || :all in measures], ",")
end


function times_to_string(times, measures)
    return join([string(to_msecs(f(skipmissing(times)))) for (x, f) in MEASURES if x in measures || :all in measures], ",")
end

function times_to_string_unchanged(times, measures)
    return join([string(f(skipmissing(times))) for (x, f) in MEASURES if x in measures || :all in measures], ",")
end

to_secs(x) = x * (10 ^ -9)
to_msecs(x) = x * (10 ^ -6)

into_time(x) = parse(Int64, x)

get_times() = map(into_time, split(readchomp(pipeline(`get_times.sh $(getpid())`, `tail -n $(Threads.nthreads())`))))



shorten(manager::CUDA.SimpleAreaManager) = "Simple($(CUDA.area_count(manager)))"
shorten(manager::CUDA.WarpAreaManager) = "Warp($(CUDA.area_count(manager)))"
shorten(poller::CUDA.AlwaysPoller) = "Const(0)"
shorten(poller::CUDA.ConstantPoller) = "Const($(Int(poller.dur)))"
shorten(poller::CUDA.VarPoller) = "Var($(length(poller.wait_durations)), $(shorten(poller.branch_predictor)))"
shorten(predictor::CUDA.SaturationCounter) = "SatCounter"
shorten(predictor::CUDA.TwoLevelPredictor) = "TwoSatCounter"
shorten(::Type{CUDA.SimpleNotificationPolicy}) = "Simple"
shorten(::Type{CUDA.TreeNotificationPolicy{N}}) where {N} = "Tree($N)"
