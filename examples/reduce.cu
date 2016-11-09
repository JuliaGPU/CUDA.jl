#include <cuda.h>
#include <stdio.h>

// Reduce a value across a warp
__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
int blockReduceSum(int val) {
  // shared mem for 32 partial sums
  static __shared__ int shared[32];

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // each warp performs partial reduction
  val = warpReduceSum(val);

  // write reduced value to shared memory
  if (lane==0) shared[wid]=val;

  // wait for all partial reductions
  __syncthreads();

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  // final reduce within first warp
  if (wid==0) {
    val = warpReduceSum_2(val);
  }

  return val;
}

__global__
void deviceReduceKernel(int *in, int* out, int N) {
  int sum = 0;

  // reduce multiple elements per thread (grid-stride loop)
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

void deviceReduce(int *in, int* out, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceKernel<<<blocks, threads>>>(in, out, N);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

const size_t len = 123456;

int main() {
    int a[len];
    for (int i = 0; i < len; i++)
        a[i] = 1;

    int *gpu_a, *gpu_b;
    cudaMalloc(&gpu_a, len*sizeof(int));
    cudaMemcpy(gpu_a, a, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_b, len*sizeof(int));

    deviceReduce(gpu_a, gpu_b, len);

    int b[len];
    cudaMemcpy(b, gpu_b, len*sizeof(int), cudaMemcpyDeviceToHost);
}
