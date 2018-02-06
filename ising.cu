#include <time.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "include/hpc_helpers.hpp"
#include "include/binary_IO.hpp"
#include "include/bitmap_IO.hpp"
#include <iostream>
#include <fstream>

#define LENGTH 1<<8 // 4096, 1<<15 works too, 1024 gives weird results
#define BLOCKSQRT 11 // 
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
#define N_TEMPS 26 // The number of increases in temperature // default 26
#define DELTA_T 0.1 // The amount of Kelvin to increase for every N_TEMPS // default 0.1
#define RUNS 5 // Reduce the lattice size by half // 7
#define ITERATIONS 4 // Iterations within a window before moving on 100 // default 4
#define OVERALL_ITERATIONS 128 // ((LENGTH * 2 + THREAD_TILE_WIDTH - 1) / THREAD_TILE_WIDTH) // Iterations of all blocks // default 128, is multiplied with length*length/2 later
#define SLIDING_ITERATIONS 32 // ((LENGTH * 2 + THREAD_TILE_WIDTH + 15) / (THREAD_TILE_WIDTH + 16)) // Sliding window within a block (left to right) for one circle // default 8
// the stride of the windows is THREAD_TILE_WIDTH/2
#define RUNS_AVERAGE 1 // Runs to calculate the average (just to be sure) // default 1
#define DATA_PER_RUN 100 // Number of times where the current state should be flushed to disk // default 100, max 131072
#define RUNS_BEFORE_FLUSH 100 // Number of iterations between two flushes // default 100
#define DUMP_IMG 0 // Set to 0 if you don't want any images
#define MC 0 // Use this for data on xmgrace

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
        int to_shuffle = THREAD_TILE_WIDTH + 1 
            + (i+black2)*(THREAD_TILE_WIDTH+2);
        float new_pad = __shfl_down(register_spins[to_shuffle], 1);
        int shuffle_here = THREAD_TILE_WIDTH 
            + (i+black2)*(THREAD_TILE_WIDTH+2);
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

__global__ void initialise_curand_on_kernels(curandState * state, 
    unsigned long seed)
{
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ void update_spins(float * register_spins, bool black, 
    curandState* globalState, float temperature)
{   
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    // Update first row
    #pragma unroll
    for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
    {
        int i = THREAD_TILE_WIDTH+1 + !black + 2*col;
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
        if(energy < 0)
        {
            register_spins[i] *= (-1);
        } else 
        {
            float p = generate(globalState, idx);
            if(p < expf(-temperature*2*energy))
            {
                register_spins[i] *= (-1);
            }  
        }
    }
    // Update intermediate
    #pragma unroll
    for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
    {
        int i = 2*THREAD_TILE_WIDTH+3 + 2*col + black;
        // Check boundaries
        float energy = register_spins[i-THREAD_TILE_WIDTH-2];
        energy += register_spins[i+THREAD_TILE_WIDTH+2];
        energy += register_spins[i-1];
        energy += register_spins[i+1];
        energy *= register_spins[i];
        if(energy < 0)
        {
            register_spins[i] *= (-1);
        } else 
        {
            float p = generate(globalState, idx);
            if( p < expf(-temperature*2*energy))
            {
                register_spins[i] *= (-1);
            }
        }
    }
    
    #pragma unroll
    for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
    {
        int i = 2*THREAD_TILE_WIDTH+3 + THREAD_TILE_WIDTH+2 + 2*col + !black;
        // Check boundaries
        float energy = register_spins[i-THREAD_TILE_WIDTH-2];
        energy += register_spins[i+THREAD_TILE_WIDTH+2];
        energy += register_spins[i-1];
        energy += register_spins[i+1];
        energy *= register_spins[i];
        if(energy < 0)
        {
            register_spins[i] *= (-1);
        } else 
        {
            float p = generate(globalState, idx);
            if( p < expf(-temperature*2*energy))
            {
                register_spins[i] *= (-1);
            }
        }
    }
    
    // Update bottom
    #pragma unroll
    for(int col=0; col<THREAD_TILE_WIDTH/2; col++)
    {
        int i = 2*THREAD_TILE_WIDTH+3 
            + THREAD_TILE_WIDTH/2*(THREAD_TILE_WIDTH+2) + 2*col + black;
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
            if( p < expf(-temperature*2*energy))
            {
                register_spins[i] *= (-1);
            }
        }
    }
}

__global__ void isis(float * spins, int length, curandState* globalState, 
    int sy, float temperature)
{  
    float register_spins[REGISTER_SIZE];
    // Use a sliding window 
    for(int sx=0; sx < SLIDING_ITERATIONS; sx++)
    {
        // Each block processes THREAD_TILE_WIDTH x THREAD_TILE_WIDTH as many values hence the THREAD_TILE_WIDTH
        int idx_x_global = threadIdx.x * THREAD_TILE_WIDTH 
            + THREAD_TILE_WIDTH*blockDim.x * blockIdx.x
            + THREAD_TILE_WIDTH*3/2 * sx;
        // Basically the row of the model
        int idx_y_global = THREAD_TILE_WIDTH*blockDim.y * blockIdx.y
            + THREAD_TILE_WIDTH/2 * sy;
        // Check if thread starts within the model
        if(idx_x_global < length && idx_y_global < length) 
        {
            // Load values to register. Each thread handles a tile of 4*4 values
            // with 4*4 padded values. Remember: Shuffle boundaries!
            // Upper boundary
            #pragma unroll
            for(int x=0; x < THREAD_TILE_WIDTH; x++)
            {
                int x_spin_idx = (idx_x_global + x)%length;
                int y_spin_idx = (idx_y_global - 1) * length;
                if(y_spin_idx < 0) y_spin_idx = length * (length-1); 
                register_spins[x] = spins[x_spin_idx + y_spin_idx];
            }
            // Middle part + boundaries left and right
            #pragma unroll
            for(int y=0; y < THREAD_TILE_WIDTH; y++)
            {
                for(int x=-1; x < THREAD_TILE_WIDTH+1; x++)
                {
                    int x_spin_idx = (idx_x_global + x)%length;
                    int y_spin_idx = (y + idx_y_global)%length * length;
                    int idx_register = 5 + x + y*(THREAD_TILE_WIDTH+2);
                    if(x_spin_idx < 0) x_spin_idx += length;
                    register_spins[idx_register] = 
                        spins[x_spin_idx + y_spin_idx];
                }
            }
            // Bottom boundary
            #pragma unroll
            for(int x=0; x < THREAD_TILE_WIDTH; x++)
            {
                int x_spin_idx = (idx_x_global + x)%length;
                int y_spin_idx = (idx_y_global + 
                                  THREAD_TILE_WIDTH)%length * length;
                register_spins[x] = spins[x_spin_idx + y_spin_idx];
            }
            
            bool black = 0;
            // Update the spins
            #pragma unroll
            for(int i=0; i < ITERATIONS; i++)
            {
                update_spins(register_spins, black, globalState, temperature);
                shuffle_spins(register_spins, black);
                black = !black;
            }
            
            // write back to spins
            #pragma unroll
            for(int y=0; y < THREAD_TILE_WIDTH; y++)
            {
                for(int x=0; x < THREAD_TILE_WIDTH; x++)
                {
                    int x_spin_idx = (idx_x_global + x)%length;
                    int y_spin_idx = (y + idx_y_global)%length * length;
                    int idx_register = 5 + x + y*(THREAD_TILE_WIDTH+2);
                    spins[x_spin_idx + y_spin_idx] = 
                        register_spins[idx_register];
                }
            }
        }
        __syncthreads();
    }
}

// Observables are:
// Average energy
// Average magnetization
__global__ void getObservables(float * spins, int length, float * observables)
{
    int idx_x_global = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y_global = blockIdx.y; // The y-dimension of a block is 1.
    float my_energy = 0;
    float magnetization = 0;
    
    for(int row=idx_y_global; row<length; row+=gridDim.y)
    {
        for(int col=idx_x_global; col<length; col+=blockDim.x*gridDim.x)
        {
            magnetization += spins[col + row*length];
            /*
            my_energy -= ( spins[col + row*length] 
                         * spins[(col+1)%length + row*length]
                         + spins[col + row*length] 
                         * spins[col + ((row+1)%length)*length]
                         + spins[col + row*length] 
                         * spins[(col-1)%length + row*length]
                         + spins[col + row*length] 
                         * spins[col + ((row-1)%length)*length]);
            */
            // Simplified
            int left = (col-1)%length;
            int below = (row-1)%length;
            if(left < 0) left += length;
            if(below < 0) below += length;
            my_energy -= ( spins[(col+1)%length + row*length]
                         + spins[col + ((row+1)%length)*length]
                         + spins[left + row*length]
                         + spins[col + below*length]) 
                         * spins[col + row*length] ; 
                         
        }
    }
    atomicAdd(observables, my_energy);
    atomicAdd(observables+1, magnetization);
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
    float *all_spins = nullptr;
    if(DUMP_IMG > 0)
    {
        // Save all spins for bitmaps
        double mem = 4 * LENGTH * LENGTH;
        mem *=  DUMP_IMG;
        mem /= (1024*1024);
        printf("Extra memory: %f Mbyte\n", mem);
        //float * all_spins = new float[LENGTH*LENGTH*10];
        size_t size = sizeof(float)*LENGTH*LENGTH;
        size *= DUMP_IMG;
        
        cudaMallocHost(&all_spins, size); CUERR
    }
    
    cudaMalloc(&deviceStates, sizeof(curandState)*BLOCKSIZE);             CUERR
    float * observables = nullptr;
    // We save later lattice size, temperature, energy, magnetization
    cudaMallocHost(&observables, sizeof(float)*4*RUNS*N_TEMPS);           CUERR
    float * stats = nullptr;
    cudaMallocHost(&stats, sizeof(float)*2*DATA_PER_RUN);                 CUERR
    float * Stats = nullptr;
    cudaMalloc(&Stats, sizeof(float)*2*DATA_PER_RUN);                     CUERR
    float * Observables = nullptr;
    cudaMalloc(&Observables, sizeof(float)*2);                            CUERR
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
    if(grid_x < 1) grid_x = 1;
    dim3 gridDims(grid_x, grid_y, grid_z);
    TIMERSTOP(preparing)
    TIMERSTART(init_curand)
    //call curand_init on each kernel with the same random seed
    //and init the rng states
    initialise_curand_on_kernels<<<gridDims, BLOCKSIZE>>>(deviceStates,
                                                          unsigned(time(NULL))); CUERR
    TIMERSTOP(init_curand)
    TIMERSTART(H2D)
    cudaMemcpy(Spins, spins, sizeof(float)*length*length, 
               cudaMemcpyHostToDevice);                                   CUERR
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
    
    std::ofstream fs;
    if(argc > 2)
        fs.open(argv[2]);
    else
        fs.open("cuda_log");
    if(DATA_PER_RUN > 0 && !MC) 
        fs << "lattice_size\ttemperature\tenergy\tmagnetization\n";
    
    for(int run=0; run<RUNS; run++)
    {
        length = LENGTH >> run;
        grid_x = length/(THREAD_TILE_WIDTH * BLOCKSIZE);
        grid_y = length/THREAD_TILE_WIDTH;
        if(grid_x < 1) grid_x = 1;
        dim3 gridDims(grid_x, grid_y, grid_z);
        printf("Using (%d, %d, %d) blocks and %d threads per block\n", 
        grid_x, grid_y, grid_z, BLOCKSIZE);
        for(int t=0; t<N_TEMPS; t++)
        {
            float energy = 0;
            float magnet = 0;
            float temperature = DELTA_T * t + 1; 
            if(MC) temperature = 2.26915;
            float beta = 1.0/temperature;
            TIMERSTART(AVERAGES)
            for(int j=0; j<RUNS_AVERAGE; j++)
            {   
                TIMERSTART(calculating)
                
                cudaMemset(Observables, 0, sizeof(float)*2);              CUERR
                for(int i=0; i<OVERALL_ITERATIONS; i++)
                {
                    isis<<<gridDims, BLOCKSIZE>>>(Spins, 
                                                  length, 
                                                  deviceStates, 
                                                  i, 
                                                  beta);                  CUERR
                    cudaDeviceSynchronize();                              CUERR
                    
                }
                getObservables<<<gridDims, BLOCKSIZE>>>(Spins, 
                                                        length, 
                                                        Observables);     CUERR
                cudaMemcpy(&observables[t*4 + run*N_TEMPS*4 +2], 
                           Observables, sizeof(float)*2,
                           cudaMemcpyDeviceToHost);                       CUERR
                observables[t*4 + run*N_TEMPS*4] = (float) length;
                observables[t*4 + run*N_TEMPS*4 +1] = temperature;
                observables[t*4 + run*N_TEMPS*4 +2] /= length*length*2;
                observables[t*4 + run*N_TEMPS*4 +3] /= length*length;
                
                energy += observables[t*4 + run*N_TEMPS*4 + 2];
                magnet += std::fabs(observables[t*4 + run*N_TEMPS*4 + 3]);
        
                TIMERSTOP(calculating)
                // Run a few more times to store more data
                TIMERSTART(additional_data)
                cudaMemset(Stats, 0, sizeof(float)*2*DATA_PER_RUN);       CUERR
                printf("-");
                for(int k=0; k<DATA_PER_RUN; k++)
                {
                    for(int f=0; f<RUNS_BEFORE_FLUSH; f++)
                    {
                        isis<<<gridDims, BLOCKSIZE>>>(Spins, 
                                                  length, 
                                                  deviceStates, 
                                                  f, 
                                                  beta);                  CUERR
                        cudaDeviceSynchronize();                          CUERR
                    }
                    
                    getObservables<<<gridDims, BLOCKSIZE>>>(Spins, 
                                                            length, 
                                                            &Stats[k*2]); CUERR
                   
                    if(k%4 == 0) {printf("\r (>^.^)> %d     ", k); fflush(stdout);}
                    else if(k%5 == 0) {printf("\r (>°.°)> ----"); fflush(stdout);}
                    
                    ///////     Dump Image
                    if(DUMP_IMG > 0)
                    {
                        if(k%DUMP_IMG == 0 && k > 0)
                        {
                            TIMERSTART(write_states_to_disk)
                            for (uint32_t l = 0; l < DUMP_IMG; l++) {
                                int idx = k-DUMP_IMG + l;
                                std::string image_name = "imgs/movie2/ising_size_" 
                                    + std::to_string(LENGTH) +  "_" + std::to_string(idx) + ".bmp";
                                dump_bitmap(&all_spins[l*LENGTH*LENGTH], LENGTH, LENGTH, image_name);
                            }
                            TIMERSTOP(write_states_to_disk)
                        }
                        cudaMemcpy(all_spins+length*length*(k%DUMP_IMG), 
                                   Spins, sizeof(float)*length*length,
                                   cudaMemcpyDeviceToHost);                   CUERR
                   }            
                }  
                
                TIMERSTOP(additional_data)
                // Dump data to plot statistics
                if(DATA_PER_RUN > 0 && !MC) 
                {  
                    cudaMemcpy(stats, 
                               Stats, sizeof(float)*2*DATA_PER_RUN,
                               cudaMemcpyDeviceToHost);                   CUERR    
                    for(int k=0; k<DATA_PER_RUN; k++)
                    {
                        stats[k*2] /= length*length*2;
                        stats[1 + k*2] /= length*length;
                        fs << length << "\t" << temperature
                           << "\t" << stats[k*2] << "\t" 
                           << stats[1 + k*2] << "\n";
                    }
                }
                // Dump data to use with xmgrace
                if(DATA_PER_RUN > 0 && MC)
                {
                    cudaMemcpy(stats, 
                               Stats, sizeof(float)*2*DATA_PER_RUN,
                               cudaMemcpyDeviceToHost);                   CUERR
                    for(int k=0; k<DATA_PER_RUN; k++)
                    {
                        stats[k*2] /= length*length*2;
                        stats[1 + k*2] /= length*length;
                        int mc = (k+1)*RUNS_BEFORE_FLUSH;
                        fs << mc << " " << stats[1 + k*2] << "\n";
                    }
                     
                }
                       
                // Reset spins
                cudaMemcpy(Spins, spins, sizeof(float)*length*length, 
                   cudaMemcpyHostToDevice);                               CUERR   
            }
            TIMERSTOP(AVERAGES)
            observables[t*4 + run*N_TEMPS*4 + 2] = energy/RUNS_AVERAGE;
            observables[t*4 + run*N_TEMPS*4 + 3] = magnet/RUNS_AVERAGE;
            printf("Lattice %d, temperature %f, energy %f, magnetization %f\n", 
                   length, temperature, observables[t*4 + run*N_TEMPS*4 +2], 
                   observables[t*4 + run*N_TEMPS*4 +3]);
        }
        
    }
    fs.close();

    std::ofstream myfile;
    if(argc > 1)
        myfile.open(argv[1]);
    else
        myfile.open("cuda_model");
    myfile << "lattice_size\ttemperature\tenergy\tmagnetization\n";
    for(int i=0; i<RUNS; i++)
    {
        for(int j=0; j<N_TEMPS; j++)
        {
            myfile << observables[j*4 + i*N_TEMPS*4] << "\t" 
                   << observables[j*4 + i*N_TEMPS*4 + 1] 
                   << "\t" << observables[j*4 + i*N_TEMPS*4 + 2] << "\t" 
                   << observables[j*4 + i*N_TEMPS*4 + 3] << "\n";
        }
        myfile << "Next\n";
    }
    myfile.close();

    cudaFreeHost(spins);                                                  CUERR
    cudaFreeHost(observables);                                            CUERR
    cudaFreeHost(all_spins);                                              CUERR
    cudaFree(Spins);                                                      CUERR
    cudaFree(deviceStates);                                               CUERR
    cudaFree(Observables);                                                CUERR
}
