# Logging

CUDA.jl can forward diagnostics from the CUDA driver and NVIDIA libraries to Julia's
logging system. Use these messages to inspect API calls, investigate failures, and find
performance hints.

## Enabling logging

Start Julia with `JULIA_DEBUG=CUDA` to enable debug output across CUDA.jl and its libraries,
including packages loaded separately, such as cuDNN and cuTENSOR:

```sh
JULIA_DEBUG=CUDA julia
```

For less output, name individual packages. For example, `JULIA_DEBUG=cuDNN` enables cuDNN
logging, and `JULIA_DEBUG=CUDACore,cuBLAS` enables CUDA.jl's core diagnostics, the driver
log, and cuBLAS logging.

You can also toggle forwarding at run time:

```julia
cuDNN.enable_logging()
# ... calls to cuDNN ...
cuDNN.enable_logging(false)
```

This setting applies process-wide. It does not change Julia's log level: the default
logger only shows `Info` and above. To see traces, include the package in `JULIA_DEBUG`
or use a logger that accepts `Debug` messages. Library logging can be expensive even when
Julia filters out the messages, so disable forwarding in performance-sensitive code.

| Message                                  | Level   |
|:-----------------------------------------|:--------|
| Library errors and failure explanations   | `Error` |
| Warnings                                 | `Warn`  |
| Performance hints                        | `Info`  |
| API traces and kernel launches            | `Debug` |
| Driver failure explanations               | `Debug` |

The following packages provide `enable_logging`: `CUDA` (driver messages; requires a
driver supporting CUDA 12.9 or newer), `cuBLAS` (including cuBLASLt), `cuDNN`, `cuSPARSE`,
`cuTENSOR`, `cuStateVec`, and `cuTensorNet`. On Windows, cuBLAS forwarding covers cuBLASLt
only. `CUDA.enable_logging()` controls the driver; it does not enable library logging.

`cuSOLVER.enable_logging()` requires a shared library that exports NVIDIA's logging API.
If unavailable, it warns; use `CUSOLVERDN_LOG_LEVEL` as described below instead.
cuFFT, cuRAND, CUPTI, and NVML have no equivalent library log forwarding.

## Diagnosing errors

On drivers supporting CUDA 12.9 or newer, `CuError` exceptions include available driver
explanations without enabling logging:

```
CUDA error: operation not supported (code 801, ERROR_NOT_SUPPORTED)
Driver log:
  [12:34:56.789][1234][CUDA][E] ...
  [12:34:56.789][1234][CUDA][E] Returning 801 (CUDA_ERROR_NOT_SUPPORTED) from cuModuleLoadDataEx
```

`JULIA_DEBUG=CUDA` also shows driver diagnostics for failures handled internally, such as
an allocation retried after freeing memory or a driver failure inside a library. These
messages do not necessarily mean your operation failed; check its return value or exception.

## Capturing messages

Messages use the logger of the Julia task that triggered them. Messages from NVIDIA worker
threads use the global logger. Delivery is asynchronous: call `CUDA.flush_logs()` before
inspecting captured messages or closing a logger's output stream.

For example, to save cuBLAS diagnostics:

```julia
using CUDA, Logging

cuBLAS.enable_logging()
try
    open("cublas.log", "w") do io
        with_logger(SimpleLogger(io, Logging.Debug)) do
            try
                A = CUDA.rand(Float32, 16, 16)
                A * A
            finally
                CUDA.flush_logs()
            end
        end
    end
finally
    cuBLAS.enable_logging(false)
end
```

To capture NVIDIA worker-thread messages too, install a global logger with
`Logging.global_logger`. Custom loggers can select all CUDA diagnostics by their `:CUDA`
group, or individual packages by the `_module` argument of `handle_message`.

## Crash diagnostics

A crash can lose messages before Julia delivers them. For these cases, configure NVIDIA's
own output before starting Julia. Leave Julia log forwarding disabled for those libraries,
so it does not override their output settings.

| Library     | Environment variables                                  |
|:------------|:-------------------------------------------------------|
| CUDA driver | `CUDA_LOG_FILE=stderr`                                  |
| cuBLAS      | `CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stderr`           |
| cuBLASLt    | `CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=/dev/stderr`       |
| cuDNN       | `CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=stderr`            |
| cuSPARSE    | `CUSPARSE_LOG_LEVEL=5 CUSPARSE_LOG_FILE=/dev/stderr`       |
| cuSOLVER    | `CUSOLVERDN_LOG_LEVEL=5 CUSOLVERDN_LOG_FILE=/dev/stderr`   |
| cuTENSOR    | `CUTENSOR_LOG_LEVEL=5 CUTENSOR_LOG_FILE=/dev/stderr`       |
| cuQuantum   | `CUSTATEVEC_LOG_LEVEL=5`, `CUTENSORNET_LOG_LEVEL=5`        |

Replace `/dev/stderr` with a file path on systems without that device. cuQuantum's settings
above write to standard output. These messages bypass Julia's logging filters and loggers.
Log levels differ between libraries; consult NVIDIA's documentation for other settings.

## API reference

```@docs
CUDACore.enable_logging
CUDACore.flush_logs
cuBLAS.enable_logging
cuDNN.enable_logging
cuSPARSE.enable_logging
cuSOLVER.enable_logging
cuTENSOR.enable_logging
cuStateVec.enable_logging
cuTensorNet.enable_logging
```
