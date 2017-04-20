// C version

#include <stdio.h>
#include <sys/time.h>

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

const size_t ITERATIONS = 5000;

float median(float x[], int n) {
  float temp;
  int i, j;
  // the following two loops sort the array x in ascending order
  for (i = 0; i < n - 1; i++) {
    for (j = i + 1; j < n; j++) {
      if (x[j] < x[i]) {
        // swap elements
        temp = x[i];
        x[i] = x[j];
        x[j] = temp;
      }
    }
  }

  if (n % 2 == 0) {
    // if there is an even number of elements, return mean of the two elements
    // in the middle
    return ((x[n / 2] + x[n / 2 - 1]) / 2.0);
  } else {
    // else return the element in the middle
    return x[n / 2];
  }
}

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

    struct timeval cpu_t0, cpu_t1;
    gettimeofday(&cpu_t0, NULL);

    CUevent gpu_t0, gpu_t1;
    check(cuEventCreate(&gpu_t0, 0x0));
    check(cuEventCreate(&gpu_t1, 0x0));

    check(cuEventRecord(gpu_t0, NULL));

    void *args[3] = {&gpu_arr};
    check(cuLaunchKernel(fun, len, 1, 1, 1, 1, 1, 0, 0, args, 0));

    check(cuEventRecord(gpu_t1, NULL));
    check(cuEventSynchronize(gpu_t1));

    gettimeofday(&cpu_t1, NULL);

    check(cuEventElapsedTime(&gpu_time[i], gpu_t0, gpu_t1));
    gpu_time[i] *= 1000;

    cpu_time[i] = (cpu_t1.tv_sec - cpu_t0.tv_sec) +
                  (cpu_t1.tv_usec - cpu_t0.tv_usec);
  }
  check(cuProfilerStop());

  printf("CPU time: %.2fus\n", median(cpu_time, ITERATIONS));
  printf("GPU time: %.2fus\n", median(gpu_time, ITERATIONS));

  return 0;
}
