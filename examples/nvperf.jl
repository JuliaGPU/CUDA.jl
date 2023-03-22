using CUDA

NVPERF.initialize()
CUPTI.initialize_profiler()

avail = CUPTI.counter_availability()
chip = first(NVPERF.supported_chips())

me = NVPERF.CUDAMetricsEvaluator(chip, avail)

NVPERF.list_metrics(me)

m = NVPERF.Metric(me, "dram__bytes.sum.per_second")
description, unit = NVPERF.properties(m)
@show description
@show string(unit)

@show NVPERF.MetricEvalRequest(me, "dram__bytes.sum.per_second")

# Need counterDataImage
# then range then 