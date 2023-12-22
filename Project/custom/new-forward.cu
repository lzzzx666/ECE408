#define best_conv
#ifdef best_conv

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 20
using std::cout;
using std::endl;
__constant__ float Mask[5000];
__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
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
                    {
                        acc += in_4d(n, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
                    }
                }
            }
        }
        out_4d(n, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef share_2d
}

__global__ void conv_forward_kernel1(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = (H - K) + 1;
    const int W_out = (W - K) + 1;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int n = blockIdx.x, m = blockIdx.y;
    if(w < W_out && h < H_out)
    {
        float acc = 0;
        #pragma unroll
        for(int c = 0; c < C; c++)
        {
            acc += in_4d(n, c, h + 0, w + 0) * mask_4d(m, c, 0, 0) +
            in_4d(n, c, h + 0, w + 1) * mask_4d(m, c, 0, 1) +
            in_4d(n, c, h + 0, w + 2) * mask_4d(m, c, 0, 2) +
            in_4d(n, c, h + 0, w + 3) * mask_4d(m, c, 0, 3) +
            in_4d(n, c, h + 0, w + 4) * mask_4d(m, c, 0, 4) +
            in_4d(n, c, h + 0, w + 5) * mask_4d(m, c, 0, 5) +
            in_4d(n, c, h + 0, w + 6) * mask_4d(m, c, 0, 6) +
            in_4d(n, c, h + 1, w + 0) * mask_4d(m, c, 1, 0) +
            in_4d(n, c, h + 1, w + 1) * mask_4d(m, c, 1, 1) +
            in_4d(n, c, h + 1, w + 2) * mask_4d(m, c, 1, 2) +
            in_4d(n, c, h + 1, w + 3) * mask_4d(m, c, 1, 3) +
            in_4d(n, c, h + 1, w + 4) * mask_4d(m, c, 1, 4) +
            in_4d(n, c, h + 1, w + 5) * mask_4d(m, c, 1, 5) +
            in_4d(n, c, h + 1, w + 6) * mask_4d(m, c, 1, 6) +
            in_4d(n, c, h + 2, w + 0) * mask_4d(m, c, 2, 0) +
            in_4d(n, c, h + 2, w + 1) * mask_4d(m, c, 2, 1) +
            in_4d(n, c, h + 2, w + 2) * mask_4d(m, c, 2, 2) +
            in_4d(n, c, h + 2, w + 3) * mask_4d(m, c, 2, 3) +
            in_4d(n, c, h + 2, w + 4) * mask_4d(m, c, 2, 4) +
            in_4d(n, c, h + 2, w + 5) * mask_4d(m, c, 2, 5) +
            in_4d(n, c, h + 2, w + 6) * mask_4d(m, c, 2, 6) +
            in_4d(n, c, h + 3, w + 0) * mask_4d(m, c, 3, 0) +
            in_4d(n, c, h + 3, w + 1) * mask_4d(m, c, 3, 1) +
            in_4d(n, c, h + 3, w + 2) * mask_4d(m, c, 3, 2) +
            in_4d(n, c, h + 3, w + 3) * mask_4d(m, c, 3, 3) +
            in_4d(n, c, h + 3, w + 4) * mask_4d(m, c, 3, 4) +
            in_4d(n, c, h + 3, w + 5) * mask_4d(m, c, 3, 5) +
            in_4d(n, c, h + 3, w + 6) * mask_4d(m, c, 3, 6) +
            in_4d(n, c, h + 4, w + 0) * mask_4d(m, c, 4, 0) +
            in_4d(n, c, h + 4, w + 1) * mask_4d(m, c, 4, 1) +
            in_4d(n, c, h + 4, w + 2) * mask_4d(m, c, 4, 2) +
            in_4d(n, c, h + 4, w + 3) * mask_4d(m, c, 4, 3) +
            in_4d(n, c, h + 4, w + 4) * mask_4d(m, c, 4, 4) +
            in_4d(n, c, h + 4, w + 5) * mask_4d(m, c, 4, 5) +
            in_4d(n, c, h + 4, w + 6) * mask_4d(m, c, 4, 6) +
            in_4d(n, c, h + 5, w + 0) * mask_4d(m, c, 5, 0) +
            in_4d(n, c, h + 5, w + 1) * mask_4d(m, c, 5, 1) +
            in_4d(n, c, h + 5, w + 2) * mask_4d(m, c, 5, 2) +
            in_4d(n, c, h + 5, w + 3) * mask_4d(m, c, 5, 3) +
            in_4d(n, c, h + 5, w + 4) * mask_4d(m, c, 5, 4) +
            in_4d(n, c, h + 5, w + 5) * mask_4d(m, c, 5, 5) +
            in_4d(n, c, h + 5, w + 6) * mask_4d(m, c, 5, 6) +
            in_4d(n, c, h + 6, w + 0) * mask_4d(m, c, 6, 0) +
            in_4d(n, c, h + 6, w + 1) * mask_4d(m, c, 6, 1) +
            in_4d(n, c, h + 6, w + 2) * mask_4d(m, c, 6, 2) +
            in_4d(n, c, h + 6, w + 3) * mask_4d(m, c, 6, 3) +
            in_4d(n, c, h + 6, w + 4) * mask_4d(m, c, 6, 4) +
            in_4d(n, c, h + 6, w + 5) * mask_4d(m, c, 6, 5) +
            in_4d(n, c, h + 6, w + 6) * mask_4d(m, c, 6, 6);
        }
        out_4d(n, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel3(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = (H - K) + 1;
    const int W_out = (W - K) + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __shared__ float tileMatA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileMatB[TILE_WIDTH][TILE_WIDTH];
    int b = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty; 
    int column = blockIdx.x * TILE_WIDTH + tx;
    int numMatAColumns = C * K * K; 
    float acc = 0.0;
    int num_iterations = ceil(numMatAColumns / (1.0 * TILE_WIDTH));
    for (int i = 0; i < num_iterations; i++) 
    {
        int temp_col = i * TILE_WIDTH + tx, temp_row = i * TILE_WIDTH + ty;
        tileMatA[ty][tx] = 0;
        tileMatB[ty][tx] = 0;
        int W_m = row;
        int W_c = temp_col / (K * K);
        int W_h = (temp_col % (K * K)) / K, W_w = (temp_col % (K * K)) % K;
 
        if (temp_col < numMatAColumns && row < M)
            tileMatA[ty][tx] = mask_4d(W_m, W_c, W_h, W_w);
        else
            tileMatA[ty][tx] = 0;
        int X_b = b;
        int X_c = temp_row / (K * K);
        int X_p = temp_row % (K * K) / K , X_q = (temp_row % (K * K)) % K;
        int X_h = column / W_out, X_w = column % W_out;

        if (temp_row < numMatAColumns && column < H_out * W_out)
            tileMatB[ty][tx] = in_4d(X_b, X_c, X_h + X_p, X_w + X_q);
        else
            tileMatB[ty][tx] = 0;

        __syncthreads();

        for (int q = 0; q < TILE_WIDTH; q++)
        {
            acc += tileMatA[ty][q] * tileMatB[q][tx];            
        }
        __syncthreads();
        int Y_b = b;
        int Y_m = row;
        int Y_h = column / W_out, Y_w = column % W_out;

        if (row < M && column < W_out * H_out)
            out_4d(Y_b, Y_m, Y_h, Y_w) = acc;
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
    cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, C * M * K * K * sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, C * M * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mask, host_mask, C * M * K * K * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    std::cout<<"m = "<<M<<std::endl;  
    if(S == 1 && K == 7) 
    {
        const int H_out = (H - K) + 1;
        const int W_out = (W - K) + 1;
        int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
        int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
        int Y = H_grid * W_grid;
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 DimGrid(B, M, Y);
        conv_forward_kernel1<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K);
        // const int H_out = (H - K) + 1;
        // const int W_out = (W - K) + 1;
        // dim3 DimGrid(ceil(H_out * W_out / (1.0 * TILE_WIDTH)), ceil(M / (1.0 * TILE_WIDTH)), B);
        // dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        // conv_forward_kernel2<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K);
    }
    // Set the kernel dimensions and call the kernel
    else
    {
        const int H_out = (H - K)/S + 1;
        const int W_out = (W - K)/S + 1;
        int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
        int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
        int Y = H_grid * W_grid;
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 DimGrid(B, M, Y);
        conv_forward_kernel<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    }
  
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask); 
    // Free device memory
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