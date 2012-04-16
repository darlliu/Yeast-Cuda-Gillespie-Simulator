#include "cuda.h"
#include <iostream>
#include <fstream>
using namespace std;

#define BDIM 12
#define BLOCKS 2
#define LOOPS 10

int __device__ get_coord()
        {

            // assume a 1 dimensional thread array
            const int tx = threadIdx.x; // thread index in thread-block (0-indexed)
            const int bx = blockIdx.x;  // block index (0-indexed)
            const int bdim = blockDim.x; 
            //const int gx = gridDim.x;

            // logical single index for this thread
            int n = tx + bdim*bx;	        
            return n;
        };

__constant__ unsigned int shift1[4] = {6, 2, 13, 3};
__constant__ unsigned int shift2[4] = {13, 27, 21, 12};
__constant__ unsigned int shift3[4] = {18, 2, 7, 13};
__constant__ unsigned int offset[4] = {4294967294, 4294967288, 4294967280, 4294967168};

__shared__ unsigned int randStates[BDIM];

__device__ unsigned int TausStep(unsigned int &z, int S1, int S2, int S3, unsigned int M)
{
        unsigned int b = (((z << S1) ^ z) >> S2);
        return z = (((z &M) << S3) ^ b);
}

__device__ unsigned int randInt()
{
        TausStep(randStates[threadIdx.x&(BDIM-1)], shift1[threadIdx.x&3], shift2[threadIdx.x&3],shift3[threadIdx.x&3],offset[threadIdx.x&3]);
        return (randStates[(threadIdx.x)&(BDIM-1)]^randStates[(threadIdx.x+1)&(BDIM-1)]^randStates[(threadIdx.x+2)&(BDIM-1)]^randStates[(threadIdx.x+3)&(BDIM-1)]);
}


//above stolen ran num generator from zarnick

template <const unsigned int OFFSET>\
void __global__ testrun (unsigned int * dev_Ran, int loop)
{
    int n= get_coord();
    TausStep(randStates[threadIdx.x&(BDIM-1)], shift1[threadIdx.x&3], shift2[threadIdx.x&3],shift3[threadIdx.x&3],offset[threadIdx.x&3]);
    __syncthreads();
    unsigned int out = (randStates[(threadIdx.x)&(BDIM-1)]^randStates[(threadIdx.x+1)&(BDIM-1)]^randStates[(threadIdx.x+2)&(BDIM-1)]^randStates[(threadIdx.x+3)&(BDIM-1)]);
    __syncthreads();
    dev_Ran[n+loop*OFFSET]=out;
};

template <const unsigned int loops, const unsigned int offset >\
void __global__ init (unsigned int * dev_Ran)
{
    int n= get_coord();
    {
        for (int i=0; i < loops ; i++)
            dev_Ran[i*offset +n]=0;
    }
};
