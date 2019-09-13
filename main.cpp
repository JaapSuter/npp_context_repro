
#include <vector>
#include <cstdio>
#include <cassert>

#include "cuda_runtime.h"
#include "npp.h"

#define ENSURE(expr) do { if (expr) break; std::printf("Error: %s\n", #expr); std::abort(); } while (false)

int main() {

    // Toggle this variable to `true` to make the test fail.
    const bool useCudaStream = false;

    const size_t width = 32768;
    const size_t height = 32768;

    std::vector<uint16_t> hostData;
    hostData.resize(width * height);

    void* deviceData = nullptr;
    size_t devicePitch = 0;
    cudaError_t cudaResult = cudaMallocPitch(&deviceData, &devicePitch, width * sizeof(uint16_t), height);
    ENSURE(cudaSuccess == cudaResult);

    cudaStream_t cudaStream = nullptr;
    NppStreamContext nppStreamContext{};
    NppStatus nppStatus = NPP_SUCCESS;
    
    if (useCudaStream) {
        // Create our own stream, make it non-blocking with stream 0.        
        cudaResult = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
        ENSURE(cudaSuccess == cudaResult);

        // Create an NPP stream context that uses this stream.
        nppStatus = nppGetStreamContext(&nppStreamContext);
        ENSURE(NPP_SUCCESS == nppStatus);
        nppStreamContext.hStream = cudaStream;
    }
    else {
        // Get the default NPP stream.
        cudaStream = nppGetStream();
    }

    const int runNumTimes = 10;
    for (int run = 0; run < runNumTimes; ++run) {

        std::printf("Run: %d\n", run);

        for (size_t i = 0; i < hostData.size(); ++i)
            hostData[i] = 0xFFFF;

        const size_t hostPitch = width * sizeof(uint16_t);
        const size_t byteWidth = width * sizeof(uint16_t);
        cudaResult = cudaMemcpy2D(deviceData, devicePitch, hostData.data(), hostPitch, byteWidth, height, cudaMemcpyDefault);
        ENSURE(cudaSuccess == cudaResult);

        const int shiftNumBits = 6;

        if (useCudaStream) {
            nppStatus = nppiRShiftC_16u_C1IR_Ctx(shiftNumBits, static_cast<Npp16u*>(deviceData), int(devicePitch), NppiSize{ int(width), int(height) }, nppStreamContext);
            ENSURE(NPP_SUCCESS == nppStatus);            
        }
        else {
            nppStatus = nppiRShiftC_16u_C1IR(shiftNumBits, static_cast<Npp16u*>(deviceData), int(devicePitch), NppiSize{ int(width), int(height) });
            ENSURE(NPP_SUCCESS == nppStatus);
        }

        cudaResult = cudaMemcpy2DAsync(hostData.data(), hostPitch, deviceData, devicePitch, byteWidth, height, cudaMemcpyDefault, cudaStream);
        ENSURE(cudaSuccess == cudaResult);

        cudaResult = cudaStreamSynchronize(cudaStream);
        ENSURE(cudaSuccess == cudaResult);

        cudaDeviceSynchronize();

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const int expected = 0xFFFF >> shiftNumBits;
                const int actual = hostData[y * width + x];
                if (expected != actual) {
                    std::printf("Error at (x, y) = (%d, %d): expected %d != actual %d\n", int(x), int(y), expected, actual);
                }
                ENSURE(expected == actual);
            }
        }
    }
    
    return 0;    
}
