#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Total Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "Max Registers per SM: " << prop.regsPerMultiprocessor << std::endl;

    return 0;
}