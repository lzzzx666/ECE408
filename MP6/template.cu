// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *aux, bool flag) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[2 * BLOCK_SIZE]; 
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) XY[threadIdx.x] = input[i];
  else XY[threadIdx.x] = 0;
  if (i + BLOCK_SIZE < len) XY[threadIdx.x + BLOCK_SIZE] = input[i + BLOCK_SIZE];
  else XY[threadIdx.x + BLOCK_SIZE] = 0;
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
  { 
    __syncthreads(); 
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < 2 * BLOCK_SIZE && index >= stride)
    { 
      XY[index] += XY[index - stride];
    } 
  } 
  //post scan step
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
  { 
    __syncthreads(); 
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < 2 * BLOCK_SIZE) 
    { 
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads(); 
  if (i < len) output[i] = XY[threadIdx.x];
  if (i + BLOCK_SIZE < len) output[i + BLOCK_SIZE] = XY[threadIdx.x + BLOCK_SIZE];
  // __syncthreads();
  if (flag) // if flag is ture, we change the aux array
  {
    if (threadIdx.x == BLOCK_SIZE - 1)
    aux[blockIdx.x] = XY[2 * BLOCK_SIZE - 1];
  }
 
}

__global__ void scan_helper(float *output, int len, float *aux)
{
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len && blockIdx.x >= 1) output[i] += aux[blockIdx.x - 1];
  if(i + BLOCK_SIZE < len && blockIdx.x >= 1)  output[i + BLOCK_SIZE] += aux[blockIdx.x - 1];
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *aux;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&aux, ceil(numElements / (BLOCK_SIZE * 2.0)) * sizeof(float))); //allocate memory for the aux array
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(aux, 0, ceil(numElements / (BLOCK_SIZE * 2.0)) * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 Dimgrid(ceil(numElements / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 Dimblock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<Dimgrid, Dimblock>>>(deviceInput, deviceOutput, numElements, aux, true); //first scan
  dim3 Dimgrid2(1, 1, 1);
  scan<<<Dimgrid2, Dimblock>>>(aux, aux, ceil(numElements / (2.0 * BLOCK_SIZE)), aux, false);// scan for block sum
  scan_helper<<<Dimgrid, Dimblock>>>(deviceOutput, numElements, aux);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
