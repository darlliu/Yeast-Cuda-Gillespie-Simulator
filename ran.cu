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
            return (float)(seed[0])/4294967296;
        case 1:
            return (float)(seed[1])/4294967296;
        case 2:
            return (float)(seed[2])/4294967296;
        case 3:
            return (float)(seed[3])/4294967296/5+0.9;
        case 4:
            return (float)(seed[4])/4294967296/33+0.04; 
            //here is the stationary period constant
        case 5:
            return (float)(seed[5])/4294967296/10+0.55;
            // here returns the 
        default:
            return (float)seed[0];
    }
};


