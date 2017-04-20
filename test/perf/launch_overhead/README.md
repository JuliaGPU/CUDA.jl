Launch overhead measurement
===========================

These tests allow measuring the overhead of launching a kernel, and comparing it to CUDA.

Use `nvvp` (the NVIDIA visual profiler) to visualize the overhead, disabling the option
"Start execution with profiling enabled".

For example:

```
$ nvprof --profile-from-start off ./cuda
==9929== NVPROF is profiling process 9929, command: ./cuda
CPU time: 36.00us
GPU time: 30.82us
==9929== Profiling application: ./cuda
==9929== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  125.70us         5  25.139us  25.088us  25.281us  kernel_dummy
```

This shows how launching a kernel takes 36us from Julia's POV, 30 us when using event
counters, but even that contains some overhead because according to `nvprof` the kernel only
took 25 us.

Luckily, this was using CUDA, and CUDAdrv.jl doesn't perform much worse:

```
$ nvprof --profile-from-start off ./cuda.jl
==19694== NVPROF is profiling process 19694, command: julia ./cuda.jl
CPU time: 36.23us
GPU time: 31.62us
==19694== Profiling application: julia ./cuda.jl
==19694== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  125.70us         5  25.139us  25.088us  25.312us  kernel_dummy
```

But more importantly, CUDAnative.jl performs equally well:

```
$ nvprof --profile-from-start off ./cudanative.jl
==21135== NVPROF is profiling process 21135, command: julia ./cudanative.jl
CPU time: 36.42us
GPU time: 31.81us
==21135== Profiling application: julia ./cudanative.jl
==21135== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  123.78us         5  24.755us  24.704us  24.928us  julia_kernel_dummy_60488
```

Note that these are simple kernels, with more complex kernels Julia's heuristics start
fighting us (eg. when dealing with long argument lists, inference performs worse and
sometimes refuses to expand our generated functions).

Also, when dealing with more arguments there's an overhead caused by CUDA copying over
arguments, and cannot be avoided. For use of hardware counters, see the CUPTI library.
