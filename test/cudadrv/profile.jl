@test_logs (:warn, r"only informs an external profiler to start") CUDA.Profile.start()
CUDA.Profile.stop()

@test_logs (:warn, r"only informs an external profiler to start") CUDA.@profile begin end
