#include <time.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "include/hpc_helpers.hpp"
#include "include/binary_IO.hpp"
#include "include/bitmap_IO.hpp"
#include <iostream>

#define BLOCKSIZE 1024
// Given n threads
// Each spin needs 4 neighbours -> padding of 4*2*sqrt(n) for each side
// e.g. n = 1024 threads
// Checkerboard update in quadratic manner -> sqrt(n) updates on each side
// Twice because of checkerboard and times 4 for each side
// 256 padded values + n*2 (checkerboard) for actual updates
// 2304 values e.g. floats
// 2304 * 32 bits / (8 * 1024) = 9 KB (fits nicely in shared memory)
// TODO: Create a register variant with warp shuffle after each update
// Register size: 256 KB / n = 64 floats for each thread
// Each thread needs in checkerboard only 32 values where we divide
// the grid within each block in tiles in ch
#define BLOCKSQRT 32

#define REGISTER_SIZE 32
#define THREAD_TILE_WIDTH 4
#define N_UPDATE (THREAD_TILE_WIDTH*THREAD_TILE_WIDTH/2)
#define TEMPERATURE 3.0
#define ITERATIONS 10000
#define OVERALL_ITERATIONS 1000

__device__ void shuffle_spins(float * register_spins, bool black)
{
    
    // Shuffle up right hand side
    #pragma unroll
    for(int i=0; i < THREAD_TILE_WIDTH; i+=2)
    {
        int to_shuffle = THREAD_TILE_WIDTH*2 + (i+black)*(THREAD_TILE_WIDTH+2);
        float new_pad = __shfl_up(register_spins[to_shuffle], 1);
        int shuffle_here = THREAD_TILE_WIDTH + (i+black)*(THREAD_TILE_WIDTH+2);
        register_spins[shuffle_here] = new_pad;
    }
    
    // Shuffle down left hand side
    bool black2 = !black;
    #pragma unroll
    for(int i=0; i < THREAD_TILE_WIDTH; i+=2)
    {
        int to_shuffle = THREAD_TILE_WIDTH + 1 + (i+black2)*(THREAD_TILE_WIDTH+2);
        float new_pad = __shfl_down(register_spins[to_shuffle], 1);
        int shuffle_here = THREAD_TILE_WIDTH + (i+black2)*(THREAD_TILE_WIDTH+2);
        register_spins[shuffle_here] = new_pad;
    }
}

__device__ float generate(curandState* globalState, int ind)
{
    //copy state to local mem
    curandState localState = globalState[ind];
    //apply uniform distribution with calculated random
    float rndval = curand_uniform( &localState );
    //update state
    globalState[ind] = localState;
    //return value
    return rndval;
}

__global__ void initialise_curand_on_kernels(curandState * state, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x 
                + threadIdx.y * blockDim.x 
                + blockIdx.y * gridDim.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ void update_spins(float * register_spins, bool black, curandState* globalState)
{   
    int idx = threadIdx.x + blockIdx.x*blockDim.x 
                + threadIdx.y * blockDim.x 
                + blockIdx.y * gridDim.x * blockDim.x;
    // Update first row
    #pragma unroll
    for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
    {
        int i = THREAD_TILE_WIDTH+1 + black + 2*col;
        // Check boundaries
        // up
        float energy_before = register_spins[i-THREAD_TILE_WIDTH-1];
        // down
        energy_before += register_spins[i+THREAD_TILE_WIDTH+2];
        // left
        energy_before += register_spins[i-1];
        // right
        energy_before += register_spins[i+1];
        float energy_after = energy_before * (-1);
        energy_before *= (-1) * register_spins[i];
        // energy_after was multiplied with (-1) before, hence again.
        energy_after *= (-1) * register_spins[i];
        if(energy_after < energy_before)
        {
            register_spins[i] *= (-1);
        } else 
        {
            float p = generate(globalState, idx);
            if( p < expf(-TEMPERATURE*(energy_after - energy_before)))
            {
                register_spins[i] *= (-1);
            }
            
        }
    }
        
    // Update intermediate
    for(int j=0; j<THREAD_TILE_WIDTH/2; j++)
    {
        #pragma unroll
        for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
        {
            int i = 2*THREAD_TILE_WIDTH+3 + j*(THREAD_TILE_WIDTH+2) + 2*col + black;
            // Check boundaries
            float energy_before = register_spins[i-THREAD_TILE_WIDTH-2];
            energy_before += register_spins[i+THREAD_TILE_WIDTH+2];
            energy_before += register_spins[i-1];
            energy_before += register_spins[i+1];
            float energy_after = energy_before * (-1);
            energy_before *= (-1) * register_spins[i];
            // energy_after was multiplied with (-1) before, hence again.
            energy_after *= (-1) * register_spins[i];
            if(energy_after < energy_before)
            {
                register_spins[i] *= (-1);
            } else 
            {
                float p = generate(globalState, idx);
                if( p < expf(-TEMPERATURE*(energy_after - energy_before)))
                {
                    register_spins[i] *= (-1);
                }
                
            }
        }
    }
    
    // Update bottom
    #pragma unroll
    for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
    {
        int i = 2*THREAD_TILE_WIDTH+3 + THREAD_TILE_WIDTH/2*(THREAD_TILE_WIDTH+2) + 2*col + black;
        // Check boundaries
        float energy_before = register_spins[i-THREAD_TILE_WIDTH-2];
        energy_before += register_spins[i+THREAD_TILE_WIDTH+1];
        energy_before += register_spins[i-1];
        energy_before += register_spins[i+1];
        float energy_after = energy_before * (-1);
        energy_before *= (-1) * register_spins[i];
        // energy_after was multiplied with (-1) before, hence again.
        energy_after *= (-1) * register_spins[i];
        if(energy_after < energy_before)
        {
            register_spins[i] *= (-1);
        } else 
        {
            float p = generate(globalState, idx);
            if( p < expf(-TEMPERATURE*(energy_after - energy_before)))
            {
                register_spins[i] *= (-1);
            }
        }
    }
}

__global__ void isis(float * spins, int length, curandState* globalState)
{
    // Each block processes twice as many values hence the 2
    int idx_x_global = threadIdx.x + 2*blockDim.x * blockIdx.x;
    int idx_y_global = threadIdx.y + 2*blockDim.y * blockIdx.y;
    
    float register_spins[REGISTER_SIZE];
    // Load values to register. Each thread handles a tile of 4*4 values
    // with 4*4 padded values. Remember: Shuffle boundaries!
    #pragma unroll
    for(int x=0; x < THREAD_TILE_WIDTH; x++)
    {
        int x_spin_idx = (idx_x_global + x)%length;
        int y_spin_idx = (idx_y_global - 1)%length * length;
        register_spins[x] = spins[x_spin_idx + y_spin_idx];
    }
    #pragma unroll
    for(int y=0; y < THREAD_TILE_WIDTH; y++)
    {
        for(int x=-1; x < THREAD_TILE_WIDTH+1; x++)
        {
            int x_spin_idx = (idx_x_global + x)%length;
            int y_spin_idx = (y + idx_y_global)%length * length;
            register_spins[x + (y+1)*THREAD_TILE_WIDTH] = spins[x_spin_idx + y_spin_idx];
        }
    }
    #pragma unroll
    for(int x=0; x < THREAD_TILE_WIDTH; x++)
    {
        int x_spin_idx = (idx_x_global + x)%length;
        int y_spin_idx = (idx_y_global + THREAD_TILE_WIDTH)%length * length;
        register_spins[x] = spins[x_spin_idx + y_spin_idx];
    }
    
    bool black = 0;
    // Update the spins
    #pragma unroll
    for(int i=0; i < ITERATIONS; i++)
    {
        update_spins(register_spins, black, globalState);
        shuffle_spins(register_spins, black);
        black = !black;
    }
}


int main (int argc, char * argv[]) 
{
    TIMERSTART(preparing)
    srand(time(NULL));
    //alocate space for each kernels curandState
    curandState* deviceStates;
    cudaMalloc(&deviceStates, 1*sizeof(curandState));                     CUERR
    
    int length = 4096;
    float * spins = nullptr;
    cudaMallocHost(&spins, sizeof(float)*length*length);                  CUERR
    // Generate random spins
    for(int i=0; i<length*length; i++)
    {
        int r = (rand()%2)*2 - 1;
        spins[i] = r;
    }
    
    float * Spins = nullptr;
    cudaMalloc(&Spins, sizeof(float)*length*length);                      CUERR
    // x-dimension should be 1 or else shuffle doesn't properly work.
    dim3 gridDims(length/(THREAD_TILE_WIDTH * BLOCKSIZE), length/THREAD_TILE_WIDTH, 1);
    
    //call curand_init on each kernel with the same random seed
    //and init the rng states
    initialise_curand_on_kernels<<<gridDims, BLOCKSIZE>>>(deviceStates, unsigned(time(NULL))); CUERR
    
    cudaMemcpy(Spins, spins, sizeof(float)*length*length, 
               cudaMemcpyDeviceToHost);                                     CUERR
    TIMERSTOP(preparing)
    
    TIMERSTART(calculating)
    for(int i=0; i<OVERALL_ITERATIONS; i++)
    {
        isis<<<gridDims, BLOCKSIZE>>>(Spins, length, deviceStates);         CUERR
    }
    TIMERSTOP(calculating)
    
    TIMERSTART(spins_D2H)
    cudaMemcpy(spins, Spins, sizeof(float)*length*length,
               cudaMemcpyDeviceToHost);                                     CUERR
    TIMERSTOP(spins_D2H)
    
    std::string image_name = "imgs/equilibrium.bmp";
    dump_bitmap(spins, length, length, image_name);
    
    cudaFreeHost(spins);                                                    CUERR
    cudaFree(Spins);                                                        CUERR
    cudaFree(deviceStates);                                                     CUERR
}
