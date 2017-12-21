#include <time.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "include/hpc_helpers.hpp"
#include "include/binary_IO.hpp"
#include "include/bitmap_IO.hpp"
#include <iostream>

#define LENGTH 4096 // 1024
#define BLOCKSQRT 16
#define BLOCKSIZE (BLOCKSQRT*BLOCKSQRT)
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

#define REGISTER_SIZE 32
#define THREAD_TILE_WIDTH 4 // Register size is 4*4 + padding (16 values)
#define TEMPERATURE 3.0
#define ITERATIONS 4 // Iterations within a window before moving on 100
#define OVERALL_ITERATIONS 128 // ((LENGTH * 2 + THREAD_TILE_WIDTH - 1) / THREAD_TILE_WIDTH) // Iterations of all blocks
#define SLIDING_ITERATIONS 16 // ((LENGTH * 2 + THREAD_TILE_WIDTH + 15) / (THREAD_TILE_WIDTH + 16)) // Sliding window within a block (left to right) for one circle
// the stride of the windows is THREAD_TILE_WIDTH/2

__device__ void shuffle_spins(float * register_spins, bool black)
{
    // Shuffle up right hand side
    #pragma unroll
    for(int i=0; i < THREAD_TILE_WIDTH; i+=2)
    {
        int to_shuffle = THREAD_TILE_WIDTH*2 + (i+black)*(THREAD_TILE_WIDTH+2);
        float new_pad = __shfl_up(register_spins[to_shuffle], 1);
        int shuffle_here = THREAD_TILE_WIDTH + (i+black)*(THREAD_TILE_WIDTH+2);
        if(threadIdx.x%32 != 0)
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
        if(threadIdx.x%32 != 31)
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
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ void update_spins(float * register_spins, bool black, curandState* globalState)
{   
   // int idx = threadIdx.x + blockIdx.x*blockDim.x 
   //             + threadIdx.y * blockDim.x 
   //             + blockIdx.y * gridDim.x * blockDim.x;
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    // Update first row
    #pragma unroll
    for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
    {
        int i = THREAD_TILE_WIDTH+1 + black + 2*col;
        // Check boundaries
        // up
        float energy = register_spins[i-THREAD_TILE_WIDTH-1];
        // down
        energy += register_spins[i+THREAD_TILE_WIDTH+2];
        // left
        energy += register_spins[i-1];
        // right
        energy += register_spins[i+1];
        energy *= register_spins[i];
        
        
      //  float energy_after = energy_before * (-1);
      //  energy_before *= (-1) * register_spins[i];
        // energy_after was multiplied with (-1) before, hence again.
       // energy_after *= (-1) * register_spins[i];
        if(energy < 0)
        {
            register_spins[i] *= (-1);
        } else 
        {
            float p = generate(globalState, idx);
            if(p < expf(-TEMPERATURE*2*energy))
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
            float energy = register_spins[i-THREAD_TILE_WIDTH-2];
            energy += register_spins[i+THREAD_TILE_WIDTH+2];
            energy += register_spins[i-1];
            energy += register_spins[i+1];
            energy *= register_spins[i];
            
           // float energy_after = energy_before * (-1);
           // energy_before *= (-1) * register_spins[i];
            // energy_after was multiplied with (-1) before, hence again.
           // energy_after *= (-1) * register_spins[i];
            if(energy < 0)
            {
                register_spins[i] *= (-1);
            } else 
            {
                float p = generate(globalState, idx);
                if( p < expf(-TEMPERATURE*2*energy))
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
        float energy = register_spins[i-THREAD_TILE_WIDTH-2];
        energy += register_spins[i+THREAD_TILE_WIDTH+1];
        energy += register_spins[i-1];
        energy += register_spins[i+1];
        energy *= register_spins[i];
        
        //float energy_after = energy_before * (-1);
        //energy_before *= (-1) * register_spins[i];
        // energy_after was multiplied with (-1) before, hence again.
        //energy_after *= (-1) * register_spins[i];
        if(energy < 0)
        {
            register_spins[i] *= (-1);
        } else 
        {
            float p = generate(globalState, idx);
            if( p < expf(-TEMPERATURE*2*energy))
            {
                register_spins[i] *= (-1);
            }
        }
    }
}

__global__ void isis(float * spins, int length, curandState* globalState, int sy)
{  
 //   if(   threadIdx.x + blockIdx.x*blockDim.x 
 //       + threadIdx.y * blockDim.x 
 //       + blockIdx.y * gridDim.x * blockDim.x == 0) printf("%d\n", sy);
    float register_spins[REGISTER_SIZE];
    // Use a sliding window 
    for(int sx=0; sx < SLIDING_ITERATIONS; sx++)
    {
        // Each block processes THREAD_TILE_WIDTH x THREAD_TILE_WIDTH as many values hence the THREAD_TILE_WIDTH
        int idx_x_global = threadIdx.x * THREAD_TILE_WIDTH + THREAD_TILE_WIDTH*blockDim.x * blockIdx.x
            +  THREAD_TILE_WIDTH/2 * 32 * sx;
        int idx_y_global = threadIdx.y * THREAD_TILE_WIDTH + THREAD_TILE_WIDTH*blockDim.y * blockIdx.y
            +  THREAD_TILE_WIDTH/2 * sy;
    
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
           // shuffle_spins(register_spins, black);
            black = !black;
        }
        
        // write back to spins
        #pragma unroll
        for(int y=0; y < THREAD_TILE_WIDTH; y++)
        {
            for(int x=-1; x < THREAD_TILE_WIDTH+1; x++)
            {
                int x_spin_idx = (idx_x_global + x)%length;
                int y_spin_idx = (y + idx_y_global)%length * length;
                spins[x_spin_idx + y_spin_idx] = register_spins[x + (y+1)*THREAD_TILE_WIDTH];
            }
        }
        __syncthreads();
    }
}


int main (int argc, char * argv[]) 
{
    cudaDeviceReset();
    TIMERSTART(preparing)
    srand(time(NULL));
    //alocate space for each kernels curandState
    curandState* deviceStates;
    // TODO: Make THREAD_TILE_WIDTH = length/BLOCKSIZE. Does it work? Must not be odd!
    int length = LENGTH;
    
    cudaMalloc(&deviceStates, sizeof(curandState)*BLOCKSIZE); CUERR
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
    int grid_x = length/(THREAD_TILE_WIDTH * BLOCKSIZE);
    int grid_y = length/THREAD_TILE_WIDTH;
    int grid_z = 1;
    dim3 gridDims(grid_x, grid_y, grid_z);
    TIMERSTOP(preparing)
    TIMERSTART(init_curand)
    //call curand_init on each kernel with the same random seed
    //and init the rng states
    initialise_curand_on_kernels<<<gridDims, BLOCKSIZE>>>(deviceStates, unsigned(time(NULL))); CUERR
    TIMERSTOP(init_curand)
    TIMERSTART(H2D)
    cudaMemcpy(Spins, spins, sizeof(float)*length*length, 
               cudaMemcpyHostToDevice);                                     CUERR
    TIMERSTOP(H2D)
    printf("Using (%d, %d, %d) blocks and %d threads per block\n", 
        grid_x, grid_y, grid_z, BLOCKSIZE);
    printf("Iterating %d times over all blocks and %d times within a window and %d times we slide a window\n", 
        OVERALL_ITERATIONS, ITERATIONS, SLIDING_ITERATIONS);
    uint n_updates = grid_x * grid_y * grid_z 
        * BLOCKSIZE * OVERALL_ITERATIONS * ITERATIONS * SLIDING_ITERATIONS
        * THREAD_TILE_WIDTH * THREAD_TILE_WIDTH / 4; // each thread updates in a
                                                     // checkerboard way half of its registers.
    printf("Overall %u updates on a %d x %d grid\n", n_updates, length, length);
    float used_memory = sizeof(float)*length*length + sizeof(curandState)*BLOCKSIZE;
    used_memory /= (1024*1024);
    printf("Using %f mByte data\n", used_memory);
    TIMERSTART(calculating)
    for(int i=0; i<OVERALL_ITERATIONS; i++)
    {
        isis<<<gridDims, BLOCKSIZE>>>(Spins, length, deviceStates, i);      CUERR
        cudaDeviceSynchronize();                                            CUERR
    }
    TIMERSTOP(calculating)
    
    TIMERSTART(spins_D2H)
    cudaMemcpy(spins, Spins, sizeof(float)*length*length,
               cudaMemcpyDeviceToHost);                                     CUERR
    TIMERSTOP(spins_D2H)
    TIMERSTART(dump_bmp)
    std::string image_name = "imgs/equilibrium.bmp";
    dump_bitmap(spins, length, length, image_name);
    TIMERSTOP(dump_bmp)
    cudaFreeHost(spins);                                                    CUERR
    cudaFree(Spins);                                                        CUERR
    cudaFree(deviceStates);                                                 CUERR
}
