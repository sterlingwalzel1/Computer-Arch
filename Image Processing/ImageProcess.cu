#include <stdio.h>
#include <cuda_runtime.h>

__global__ void imgProc1(int *x, int *y, int *ya, int a, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) 
        ya[i] = a * x[i] + y[i];
}

__global__ void imgProc2(int *x, int *y, int *ya, int a, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) 
        ya[i] = a * x[i] + y[i];
}

int main() {
    int n = 12;
    int a = 3;
    int *h_x, *h_y, *h_ya;
    int *d_x, *d_y, *d_ya;
    int size = n * sizeof(int);

    h_x = (int *)malloc(size);
    h_y = (int *)malloc(size);
    h_ya = (int *)malloc(size);

    for (int i = 0; i < n; i++) {
        h_x[i] = i;
        h_y[i] = i * 2;
    }

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_ya, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    imgProc<<<1, n>>>(d_x, d_y, d_ya, a, n);

    cudaMemcpy(h_ya, d_ya, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("%d * %d + %d = %d\n", a, h_x[i], h_y[i], h_ya[i]);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_ya);
    free(h_x);
    free(h_y);
    free(h_ya);

    return 0;
}