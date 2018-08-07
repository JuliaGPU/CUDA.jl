// C version

#include <stdio.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <cudaProfiler.h>

#define check(err) __check(err, __FILE__, __LINE__)
void __check(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *msg;
    cuGetErrorName(err, &msg);
    fprintf(stderr, "CUDA error: %s (%04d) at %s:%i.\n", msg, err, file, line);
    exit(-1);
  }
}

const size_t len = 1000;
const size_t ITERATIONS = 100;

int main(int argc, char **argv) {
  check(cuInit(0x0));

  CUdevice dev;
  check(cuDeviceGet(&dev, 0));

  CUcontext ctx;
  check(cuCtxCreate(&ctx, 0, dev));

  CUmodule mod;
  check(cuModuleLoad(&mod, "cuda.ptx"));

  CUfunction fun;
  check(cuModuleGetFunction(&fun, mod, "kernel_dummy"));

  CUdeviceptr gpu_arr;
  check(cuMemAlloc(&gpu_arr, sizeof(float) * len));

  float cpu_time[ITERATIONS];
  float gpu_time[ITERATIONS];

  for (int i = 0; i < ITERATIONS; i++) {
    if (i == ITERATIONS - 5)
      check(cuProfilerStart());

    struct timespec cpu_t0, cpu_t1;
    clock_gettime(CLOCK_MONOTONIC, &cpu_t0);

    CUevent gpu_t0, gpu_t1;
    check(cuEventCreate(&gpu_t0, 0x0));
    check(cuEventCreate(&gpu_t1, 0x0));

    check(cuEventRecord(gpu_t0, NULL));

    void *args[3] = {&gpu_arr};
    check(cuLaunchKernel(fun, len, 1, 1, 1, 1, 1, 0, 0, args, 0));

    check(cuEventRecord(gpu_t1, NULL));
    check(cuEventSynchronize(gpu_t1));

    clock_gettime(CLOCK_MONOTONIC, &cpu_t1);

    check(cuEventElapsedTime(&gpu_time[i], gpu_t0, gpu_t1));
    gpu_time[i] *= 1000;

    cpu_time[i] = (cpu_t1.tv_sec - cpu_t0.tv_sec) +
                  (cpu_t1.tv_nsec - cpu_t0.tv_nsec) / 1000.;
  }
  check(cuProfilerStop());

  double mean_cpu = 0;
  double mean_gpu = 0;
  int i;
  for (i = 1; i < ITERATIONS ; ++i) {
      mean_cpu += cpu_time[i];
      mean_gpu += gpu_time[i];
  }
  mean_cpu /= (ITERATIONS-1);
  mean_gpu /= (ITERATIONS-1);

  double std_cpu = 0;
  double std_gpu = 0;
  for (i = 1; i < ITERATIONS ; ++i ) {
      std_cpu += pow((cpu_time[i] - mean_cpu), 2);
      std_gpu += pow((gpu_time[i] - mean_gpu), 2);
  }
  std_cpu = sqrt(std_cpu / (ITERATIONS-1));
  std_gpu = sqrt(std_gpu / (ITERATIONS-1));

  printf("CPU time: %.2f +/- %.2f us\n", mean_cpu, std_cpu);
  printf("GPU time: %.2f +/- %.2f us\n", mean_gpu, std_gpu);

  return 0;
}
