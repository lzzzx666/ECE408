#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_SIZE 3
#define KENERL_RADIUS (KERNEL_SIZE / 2)
#define TILE_SIZE 6
#define INPUTTILE_SIZE (TILE_SIZE + KERNEL_SIZE - 1)
//@@ Define constant memory for device kernel here
__constant__ float Kernel[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size) 
{
  __shared__ float InputTile[INPUTTILE_SIZE * INPUTTILE_SIZE * INPUTTILE_SIZE];
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int x = bx * TILE_SIZE + tx, y = by * TILE_SIZE + ty, z = bz * TILE_SIZE + tz;
  int x_i = x - KENERL_RADIUS, y_i = y - KENERL_RADIUS, z_i = z - KENERL_RADIUS;

  if (x_i >= 0 && x_i < x_size && y_i >= 0 && y_i < y_size && z_i >= 0 && z_i < z_size)
    InputTile[tz * INPUTTILE_SIZE * INPUTTILE_SIZE + ty * INPUTTILE_SIZE + tx] = input[z_i * y_size * x_size  + y_i * x_size + x_i];
  else  InputTile[tz * INPUTTILE_SIZE * INPUTTILE_SIZE + ty * INPUTTILE_SIZE + tx] = 0.0;
  __syncthreads();
  float out = 0.0;
  if(tz < TILE_SIZE && ty < TILE_SIZE && tx < TILE_SIZE) 
  {
    for(int z_kernel = 0; z_kernel < KERNEL_SIZE; z_kernel++) 
    {
      for(int y_kernel = 0; y_kernel < KERNEL_SIZE; y_kernel++) 
      {
        for(int x_kernel = 0; x_kernel < KERNEL_SIZE; x_kernel++) 
        {
          out += Kernel[z_kernel * KERNEL_SIZE * KERNEL_SIZE + y_kernel * KERNEL_SIZE + x_kernel] * InputTile[(tz + z_kernel) * INPUTTILE_SIZE * INPUTTILE_SIZE + (ty + y_kernel) * INPUTTILE_SIZE + tx + x_kernel];
        }
      }
    }
    __syncthreads();
    if(z < z_size && y < y_size && x < x_size)  output[z * x_size * y_size + y * x_size + x] = out;
  }
  //@@ Insert kernel code here
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void**)&deviceInput, sizeof(float) * x_size * y_size * z_size);
  cudaMalloc((void**)&deviceOutput, sizeof(float) * x_size * y_size * z_size);
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3,  z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Kernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size / (1.0 * TILE_SIZE)), ceil(y_size / (1.0 * TILE_SIZE)), ceil(z_size / (1.0 * TILE_SIZE)));
  dim3 dimBlock(INPUTTILE_SIZE, INPUTTILE_SIZE, INPUTTILE_SIZE);
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  
  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");



  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutput + 3, deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
