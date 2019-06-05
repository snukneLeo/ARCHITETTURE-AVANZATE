#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

const int RADIUS = 7;

__global__
void stencilKernel(const int* d_input, int N,int* d_output) {
    // YOUR CODE
    
    
    int idThreads = blockIdx.x * blockDim.x * threadIdx.x;
    
    int valueArr = 0;
    
    if (idThreads > RADIUS-1 && idThread < N - RADIUS)
    {
        for (int i = 1; i <= RADIUS; i++)
        {
            valueArr += d_input[idThreads+i];
            valueArr += d_input[idThreads-i];
        }

        d_output = valueArr + d_input[idThreads];
    }
    
}

const int N  = 1000;

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_input      = new int[N];
    int* h_output_tmp = new int[N]; // <-- used for device result
    int* h_output     = new int[N](); // initilization to zero

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++)
        h_input[i] = distribution(generator);

    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for (int i = RADIUS; i < N - RADIUS; i++) {
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            h_output[i] += h_input[j];
    }

    TM_host.stop();
    TM_host.print("1DStencil host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_input, *d_output;
    SAFE_CALL( cudaMalloc(&d_input, N * sizeof(int)));
    SAFE_CALL( cudaMalloc(&d_output, N * sizeof(int)));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL(cudaMemcpy(d_input, h_input, N * sizeof(int),cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // did you miss something?
    ///
    
    dim3 block_size(128,1,1);
    dim3 num_blocks(N/128,1,1);
    if(N%128)
    {
        block_size.x++;
    }

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    stencilKernel<<< num_blocks, block_size>>>(d_input, N ,d_output);

    CHECK_CUDA_ERROR
    TM_device.stop();
    
    TM_device.print("1DStencil device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL(cudaMemcpy(h_output, d_output, N * sizeof(int),cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N; i++) {
        if (h_output[i] != h_output_tmp[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << h_output[i]
                      << "\ndevice: " << h_output_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    /// SAFE_CALL( cudaFree( ... ) )
    /// SAFE_CALL( cudaFree( ... ) )

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
