

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



void __device__ lrg ( unsigned int * seed )
{
    for (int i = 0; i < 6; i++) {
        /* code */
        seed[i] =(1103515245*seed[i]+12345)%4294967296;
    }
};

float __device__ fr  ( unsigned int * seed, unsigned int i )
{
    switch (i)
    {
        case 0:
            return ((float)seed[0])/4294967296.;
        case 1:
            return ((float)seed[1])/4294967296.;
        case 2:
            return ((float)seed[2])/4294967296.;
        case 3:
            return ((float)seed[3])/4294967296./5+0.9;
        case 4:
            return ((float)seed[4])/4294967296./50+0.03; 
            //here is the stationary period constant
        case 5:
            return ((float)seed[5])/4294967296./10+0.55;
            // here returns the 
        default:
            return (float)seed[0];
    }
};

template  <const unsigned int loops, const unsigned int offset >\
void __global__ test (unsigned int * seedpool, float * dev_Ran)
{
    unsigned int seed[6] ;
    int n=get_coord();
    for (int i = 0; i < 6; i++) {
        seed[i]=seedpool[n+i*offset];
        __syncthreads();
    }
    for (int i = 0; i < loops ; i++) {
        lrg(seed);
        dev_Ran[n+i*offset]=fr(seed, 5);
    }
}

template <const unsigned int loops, const unsigned int offset >\
void __global__ init (float * dev_Ran, unsigned int * dev_seeds)
{
    int n= get_coord();
    {
        for (int i=0; i < loops ; i++)
            dev_Ran[i*offset +n]=0;
        for (int i=0; i<6; i++)
            dev_seeds[i*offset+n]=0;
    }
};
