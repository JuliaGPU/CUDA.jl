#include <cuda.h>
#include <stdio.h>

// Reduce a value across a warp
__inline__ __device__
int sumReduce_warp(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

// Reduce a value across a block, using shared memory for communication
__inline__ __device__ int sumReduce_block(int val) {
  // shared mem for 32 partial sums
  static __shared__ int shared[32];

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // each warp performs partial reduction
  val = sumReduce_warp(val);

  // write reduced value to shared memory
  if (lane==0) shared[wid]=val;

  // wait for all partial reductions
  __syncthreads();

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  // final reduce within first warp
  if (wid==0) {
    val = sumReduce_warp(val);
  }

  return val;
}

// Reduce an array across a complete grid
__global__ void sumReduce_grid(int *input, int* output, int N) {
  int sum = 0;

  // reduce multiple elements per thread (grid-stride loop)
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += input[i];
  }

  sum = sumReduce_block(sum);

  if (threadIdx.x==0)
    output[blockIdx.x]=sum;
}

void sumReduce(int *input, int* output, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  sumReduce_grid<<<blocks, threads>>>(input, output, N);
  sumReduce_grid<<<1, 1024>>>(output, output, blocks);
}

struct State
{
  size_t len;
  int *gpu_input;
  int *gpu_output;
};

extern "C"
State *setup(int *input, size_t len)
{
  State *state = new State();

  state->len = len;

  cudaMalloc(&state->gpu_input, len*sizeof(int));
  cudaMemcpy(state->gpu_input, input, len*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&state->gpu_output, len*sizeof(int));

  return state;
}

extern "C"
int run(State *state)
{
  sumReduce(state->gpu_input, state->gpu_output, state->len);

  int output[state->len];
  cudaMemcpy(output, state->gpu_output, state->len*sizeof(int), cudaMemcpyDeviceToHost);

  return output[0];
}

extern "C"
void teardown(State *state)
{
  cudaFree(state->gpu_output);
  cudaFree(state->gpu_input);
}

const size_t len = 123456;

int main() {
  int input[len];
  for (int i = 0; i < len; i++)
      input[i] = 1;

  State *state = setup(input, len);
  int gpu_output = run(state);
  teardown(state);

  return 0;
}
