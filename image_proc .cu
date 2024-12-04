#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

// CUDA kernel for box blur filter
__global__ void boxBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;

        // Apply box filter around the pixel
        for (int blurRow = -filterWidth; blurRow <= filterWidth; ++blurRow) {
            for (int blurCol = -filterWidth; blurCol <= filterWidth; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal += input[curRow * width + curCol];
                    pixels++;
                }
            }
        }

        // Write the average value to the output pixel
        output[row * width + col] = pixVal / pixels;
    }
}

__global__ void negativeFilter(unsigned char* input, unsigned char* output, int width, int height, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get the input pixel value
        unsigned char pixVal = input[row * width + col];

        // Apply the negative filter
        unsigned char negativeVal = 255 - pixVal;

        // Write the negative value to the output pixel
        output[row * width + col] = negativeVal;
    }
}

__global__ void verticalFlipFilter(unsigned char* input, unsigned char* output, int width, int height, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Calculate the corresponding row in the flipped image
        int flippedRow = height - 1 - row;

        // Copy the pixel from the input to the output in the flipped position
        output[flippedRow * width + col] = input[row * width + col];
    }
}

__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Sobel filter kernels (3x3)
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{-1, -2, -1}, { 0,  0,  0}, { 1,  2,  1}};
        
        int gradientX = 0;
        int gradientY = 0;
        
        // Apply Sobel filter to the pixel and its neighbors
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int curRow = row + i;
                int curCol = col + j;
                
                // Check if the current pixel is within bounds
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    int pixelVal = input[curRow * width + curCol];

                    // Accumulate the weighted sum for both Gx and Gy
                    gradientX += pixelVal * Gx[i + 1][j + 1];
                    gradientY += pixelVal * Gy[i + 1][j + 1];
                }
            }
        }

        // Calculate the gradient magnitude
        int magnitude = (int)sqrtf((float)(gradientX * gradientX + gradientY * gradientY));

        // Clamp the result to the range 0-255 and store in the output
        output[row * width + col] = (unsigned char)min(magnitude, 255);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./image_blur <input_image_path>" << endl;
        return -1;
    }

    // Load image using STB Image (grayscale)
    int width, height, channels;
    unsigned char* h_inputImage1 = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!h_inputImage1) {
        cout << "Failed to load image!" << endl;
        return -1;
    }

    unsigned char* h_inputImage2 = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!h_inputImage2) {
        cout << "Failed to load image!" << endl;
        return -1;
    }

    unsigned char* h_inputImage3 = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!h_inputImage3) {
        cout << "Failed to load image!" << endl;
        return -1;
    }

    unsigned char* h_inputImage4 = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!h_inputImage4) {
        cout << "Failed to load image!" << endl;
        return -1;
    }

    int imageSize = width * height;
    unsigned char* h_outputImage1 = (unsigned char*)malloc(imageSize);
    unsigned char* h_outputImage2 = (unsigned char*)malloc(imageSize);
    unsigned char* h_outputImage3 = (unsigned char*)malloc(imageSize);
    unsigned char* h_outputImage4 = (unsigned char*)malloc(imageSize);

    // Allocate device memory
    unsigned char* d_inputImage;
    unsigned char* d_outputImage1;
    unsigned char* d_outputImage2;
    unsigned char* d_outputImage3;
    unsigned char* d_outputImage4;
    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_outputImage1, imageSize);
    cudaMalloc((void**)&d_outputImage2, imageSize);
    cudaMalloc((void**)&d_outputImage3, imageSize);
    cudaMalloc((void**)&d_outputImage4, imageSize);

    // Copy input image to device
    cudaMemcpy(d_inputImage, h_inputImage1, imageSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the box blur kernel
    int filterWidth = 1;  // Box filter size
    boxBlurKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage1, width, height, filterWidth);
    negativeFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage2, width, height, filterWidth);
    verticalFlipFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage3, width, height, filterWidth);
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage4, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_outputImage1, d_outputImage1, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputImage2, d_outputImage2, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputImage3, d_outputImage3, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputImage4, d_outputImage4, imageSize, cudaMemcpyDeviceToHost);

    // Save the output image using STB Image
    stbi_write_png("boxBlur.png", width, height, 1, h_outputImage1, width);
    stbi_write_png("negative.png", width, height, 1, h_outputImage2, width);
    stbi_write_png("verticalFlip.png", width, height, 1, h_outputImage3, width);
    stbi_write_png("sobel.png", width, height, 1, h_outputImage4, width);


    // Free memory
    stbi_image_free(h_inputImage1);
    stbi_image_free(h_inputImage2);
    stbi_image_free(h_inputImage3);
    stbi_image_free(h_inputImage4);
    free(h_outputImage1);
    free(h_outputImage2);
    free(h_outputImage3);
    free(h_outputImage4);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage1);
    cudaFree(d_outputImage2);
    cudaFree(d_outputImage3);
    cudaFree(d_outputImage4);

    cout << "Image processing complete. Output saved " << endl;
    return 0;
}
