// #define stream_me
#ifdef stream_me
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int n = blockIdx.x, m = blockIdx.y;
    if(w < W_out && h < H_out)
    {
        float acc = 0;
        for(int c = 0; c < C; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    if(h * S + p < H && w * S + q < W)
                    acc += in_4d(n, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(n, m, h, w) = acc;
    }
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, 
const float *host_input, const float *host_mask, float **device_output_ptr, 
float **device_input_ptr, float **device_mask_ptr, 
const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, M * C * K * K * sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
    int Y = H_grid * W_grid;
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(1, M, Y);
    int i;
    for(i = 0; i < B && i + 1 < B;)
    {
        int offset0 = i * C * H * W, offset1 = (i + 1) * C * H * W;
        int outset0 = i * M * H_out * W_out, outset1 = (i + 1) * M * H_out * W_out;
        cudaMemcpyAsync(*device_input_ptr + offset0, host_input + offset0, C * H * W * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(*device_input_ptr + offset1, host_input + offset1, C * H * W * sizeof(float), cudaMemcpyHostToDevice, stream1);
        conv_forward_kernel<<<DimGrid, DimBlock, 0, stream0>>>(*device_output_ptr + outset0, *device_input_ptr + offset0, *device_mask_ptr, 1, M, C, H, W, K, S);
        conv_forward_kernel<<<DimGrid, DimBlock, 0, stream1>>>(*device_output_ptr + outset1, *device_input_ptr + offset1, *device_mask_ptr, 1, M, C, H, W, K, S);
        cudaMemcpyAsync((void*)(host_output + outset0), *device_output_ptr + outset0,  M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync((void*)(host_output + outset1), *device_output_ptr + outset1, M * H_out * W_out *sizeof(float), cudaMemcpyDeviceToHost, stream1);
        i += 2;
    }    
    if(i != B)
    {
        int offset_0 = i * C * H * W;
        int outset_0 = i * M * H_out * W_out;
        cudaMemcpyAsync(*device_input_ptr + offset_0, host_input + offset_0, C * H * W * sizeof(float), cudaMemcpyHostToDevice, stream0);
        conv_forward_kernel<<<DimGrid, DimBlock, 0, stream0>>>(*device_output_ptr + outset_0, *device_input_ptr + offset_0, *device_mask_ptr, 1, M, C, H, W, K, S);
        cudaMemcpyAsync((void*)(host_output + outset_0), *device_output_ptr + outset_0,  M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost, stream0);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask); 
}
__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
#endif