NVCC=nvcc
NVCCFLAGS= -std=c++11 -D_FORCE_INLINES -arch=sm_50 -lineinfo -D_MWAITXINTRIN_H_INCLUDED

all: ising

ising: ising.cu
	$(NVCC) $(NVCCFLAGS) ising.cu -o ising

clean:
	rm -f ising  
