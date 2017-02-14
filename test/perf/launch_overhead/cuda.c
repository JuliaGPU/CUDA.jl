#include <stdio.h>
#include <sys/time.h>

#include <cuda.h>
#include <cudaProfiler.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *msg;
    cuGetErrorName(err, &msg);
    fprintf(stderr, "CUDA error: %s (%04d) at %s:%i.\n", msg, err, file, line);
    exit(-1);
  }
}

const size_t len = 100000;

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
  checkCudaErrors(cuInit(0x0));

  CUdevice dev;
  checkCudaErrors(cuDeviceGet(&dev, 0));

  CUcontext ctx;
  checkCudaErrors(cuCtxCreate(&ctx, 0, dev));

  CUmodule mod;
  checkCudaErrors(cuModuleLoad(&mod, "cuda.ptx"));

  CUfunction fun;
  checkCudaErrors(cuModuleGetFunction(&fun, mod, "kernel_dummy"));

  CUdeviceptr gpu_arr;
  checkCudaErrors(cuMemAlloc(&gpu_arr, sizeof(float) * len));

  float cpu_time[ITERATIONS];
  float gpu_time[ITERATIONS];

  for (int i = 0; i < ITERATIONS; i++) {
    if (i == ITERATIONS - 5)
      checkCudaErrors(cuProfilerStart());

    struct timeval cpu_t0, cpu_t1;
    gettimeofday(&cpu_t0, NULL);

    CUevent gpu_t0, gpu_t1;
    checkCudaErrors(cuEventCreate(&gpu_t0, 0x0));
    checkCudaErrors(cuEventCreate(&gpu_t1, 0x0));

    checkCudaErrors(cuEventRecord(gpu_t0, NULL));

    void *args[3] = {&gpu_arr};
    checkCudaErrors(cuLaunchKernel(fun, len, 1, 1, 1, 1, 1, 0, 0, args, 0));

    checkCudaErrors(cuEventRecord(gpu_t1, NULL));
    checkCudaErrors(cuEventSynchronize(gpu_t1));

    checkCudaErrors(cuEventElapsedTime(&gpu_time[i], gpu_t0, gpu_t1));

    gettimeofday(&cpu_t1, NULL);
    cpu_time[i] = (cpu_t1.tv_sec - cpu_t0.tv_sec) +
                  (cpu_t1.tv_usec - cpu_t0.tv_usec) / 1000000.0;
  }
  checkCudaErrors(cuProfilerStop());

  float overhead[ITERATIONS];
  for (int i = 0; i < ITERATIONS; i++)
    overhead[i] = 1000 * cpu_time[i] - gpu_time[i];
  printf("Overhead: %.2fus on %.2fus (%.2f%%)\n",
         1000 * median(overhead, ITERATIONS), 1000 * median(gpu_time, ITERATIONS),
         100 * median(overhead, ITERATIONS) / median(gpu_time, ITERATIONS));

  return 0;
}
