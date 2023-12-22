// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void cast_float_to_char(float* input_array, unsigned char* uchar_image, int imageWidth, int imageHeight, int imageChannels)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < imageWidth * imageHeight * imageChannels)
  {
    uchar_image[i] = (unsigned char)(255 * input_array[i]);
  }
}
__global__ void convert_RGB_to_Gray(unsigned char* uchar_image, unsigned char* gray_image, int imageWidth, int imageHeight)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < imageWidth * imageHeight)
  {
    unsigned char r = uchar_image[3 * i], g = uchar_image[3 * i + 1], b = uchar_image[3 * i + 2];
    gray_image[i] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
}
__global__ void compute_historam_gray_image(unsigned char* gray_image, unsigned int* hist, int size)
{
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  if(threadIdx.x < HISTOGRAM_LENGTH)
  {
    histo_private[threadIdx.x] = 0;
  }
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (i < size)
  {
    atomicAdd(&(histo_private[gray_image[i]]), 1);
    i += stride;
  }
  __syncthreads();
  if(threadIdx.x < HISTOGRAM_LENGTH)
  {
    atomicAdd(&(hist[threadIdx.x]), histo_private[threadIdx.x]);
  }
}
__global__ void compute_cdf(unsigned int *hist, float *cdf, int size)
{ 
  //here block_size = 128 since we only need to process 256 elements
  __shared__ float XY[HISTOGRAM_LENGTH]; 
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HISTOGRAM_LENGTH) XY[threadIdx.x] = hist[i];
  else XY[threadIdx.x] = 0;
  if (i + HISTOGRAM_LENGTH / 2 < HISTOGRAM_LENGTH) XY[threadIdx.x + HISTOGRAM_LENGTH / 2] = hist[i + HISTOGRAM_LENGTH / 2];
  else XY[threadIdx.x + HISTOGRAM_LENGTH / 2] = 0;
  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2)
  { 
    __syncthreads(); 
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < HISTOGRAM_LENGTH && index >= stride)
    { 
      XY[index] += XY[index - stride];
    } 
  } 
  //post scan step
  for (unsigned int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2)
  { 
    __syncthreads(); 
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < HISTOGRAM_LENGTH) 
    { 
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads(); 
  if (i < HISTOGRAM_LENGTH) cdf[i] = XY[threadIdx.x] / size;
  if (i + HISTOGRAM_LENGTH / 2 < HISTOGRAM_LENGTH) cdf[i + HISTOGRAM_LENGTH / 2] = XY[threadIdx.x + HISTOGRAM_LENGTH / 2] / size;
}
__global__ void equalization_function(float *cdf, unsigned char *uchar_image, float *output, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
  {
    float clamp = 255 * (cdf[uchar_image[i]] - cdf[0]) / (1 - cdf[0]);
    output[i] = (float) (min(max(clamp, 0.0), 255.0)) / 255.0;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  unsigned char *uchar_image;
  unsigned char *gray_image;
  unsigned int *hist;
  float *cdf;
  float *input_array;
  float *output_array;
  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **)&input_array, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&uchar_image, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&gray_image, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&hist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&output_array, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMemset((void *)hist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *)cdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(input_array, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);


  int grid = ceil((imageWidth * imageHeight * imageChannels) / 256.0);
  cast_float_to_char<<<grid, HISTOGRAM_LENGTH>>>(input_array, uchar_image, imageWidth, imageHeight, imageChannels);
  convert_RGB_to_Gray<<<grid, HISTOGRAM_LENGTH>>>(uchar_image, gray_image, imageWidth, imageHeight);
  compute_historam_gray_image<<<grid, HISTOGRAM_LENGTH>>>(gray_image, hist, imageWidth * imageHeight);
  compute_cdf<<<1, 128>>>(hist, cdf, imageWidth * imageHeight);
  equalization_function<<<grid, HISTOGRAM_LENGTH>>>(cdf, uchar_image, output_array, imageWidth * imageHeight * imageChannels);
  cudaMemcpy(hostOutputImageData, output_array, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);


  cudaFree(input_array), cudaFree(uchar_image), cudaFree(gray_image);
  cudaFree(hist), cudaFree(cdf), cudaFree(output_array);
  wbSolution(args, outputImage);
  wbExport("outputimg.ppm", outputImage);
  //@@ insert code here

  return 0;
}
