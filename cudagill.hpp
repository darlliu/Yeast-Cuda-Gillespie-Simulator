#include "headers.h"
void checkerror(cudaError_t err)
{
    cout <<"\tresult: ";
    switch (err)
    {
        case cudaErrorInvalidValue:
            cout<<"\twrong values "<<endl;
            throw (err);
        case cudaErrorInvalidDevicePointer:
            cout << "\tdevice pointer error"<<endl;
            throw (err);
        case cudaErrorInvalidMemcpyDirection:
            cout << "\twrong direction"<<endl;
            throw (err);
        case cudaSuccess:
            cout << "\tSuccess!!" <<endl;
            return;
        default:
            cout << "\tUnrecorded behavior: "<<err<<endl;
            throw (err);
    }
}
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

__global__ void varover(unsigned int N, unsigned int *dev_out1, \
        unsigned int *dev_out2, float *dev_temp )
{
    unsigned int n= get_coord();
    if(n<N)
    {
        dev_temp[n]=(float)dev_out1[n]/(float)dev_out2[n];
        __syncthreads();
    }
}


template <const unsigned int bdim, typename T>
__global__ void getmax(int N, T *a, T *ra){

  volatile __shared__  T s_a[bdim]; // force read from shared

  // assume a 1 dimensional thread array
  const unsigned int tx = threadIdx.x; // thread index in thread-block (0-indexed)
  const unsigned int bx = blockIdx.x;  // block index (0-indexed)
  const unsigned int gx = gridDim.x;

  // logical single index for this thread
  unsigned int n = tx + bdim*bx;

  // is thread in range for the addition
  float d = 0.f;
  while(n<N){
    d = (d>a[n])?d:a[n];
    n += gx*bdim;
  }
  
  // assume bx power of 2 
  s_a[tx] = d;
  __syncthreads();


  // unroll
  if(bdim>=1024){
    if(tx<512) s_a[tx] = (s_a[tx]>s_a[tx+512])?s_a[tx]:s_a[tx+512]; 
    __syncthreads();
  }  	       
  if(bdim>=512){
    if(tx<256) s_a[tx] = (s_a[tx]>s_a[tx+256])?s_a[tx]:s_a[tx+256]; 
    __syncthreads();
  }  	       
  if(bdim>=256){
    if(tx<128) s_a[tx]= (s_a[tx]>s_a[tx+128])?s_a[tx]:s_a[tx+128]; 
    __syncthreads();
  }
  if(bdim>=128){
    if(tx<64)  s_a[tx] = (s_a[tx]>s_a[tx+64])?s_a[tx]:s_a[tx+64]; 
    __syncthreads();
  }
  if(tx<32){
    s_a[tx] = (s_a[tx]>s_a[tx+32])?s_a[tx]:s_a[tx+32];
    s_a[tx] = (s_a[tx]>s_a[tx+16])?s_a[tx]:s_a[tx+16];
    s_a[tx] = (s_a[tx]>s_a[tx+8])?s_a[tx]:s_a[tx+8];
    s_a[tx] = (s_a[tx]>s_a[tx+4])?s_a[tx]:s_a[tx+4];
    s_a[tx] = (s_a[tx]>s_a[tx+2])?s_a[tx]:s_a[tx+2];
  }

  // final reduction
  if(tx==0)
    ra[bx] = (s_a[0] > s_a[1])?s_a[0]:s_a[1];
}

template <const unsigned int bdim, unsigned int loops, typename T, typename S>
__global__ void variance (S *a, T *ra, T *mean){
//note that the bdim should be loop while bx should be number of elements
  volatile __shared__  T s_a[bdim]; // force read from shared

  // assume a 1 dimensional thread array
  const unsigned int tx = threadIdx.x; // thread index in thread-block (0-indexed)
  const unsigned int bx = blockIdx.x;  // block index (0-indexed)

  // logical single index for this thread
  unsigned int n = tx ;

  // is thread in range for the addition
  T d = 0.f;
  while(n<loops){
    d += ((T)a[n+bx*loops]-mean[bx])*((T)a[n+bx*loops]-mean[bx]);
    n += bdim;
  }
  
  // assume bx power of 2 
  s_a[tx] = d;
  __syncthreads();


  // unroll
  if(bdim>=1024){
    if(tx<512) s_a[tx] += s_a[tx+512]; 
    __syncthreads();
  }  	       
  if(bdim>=512){
    if(tx<256) s_a[tx] += s_a[tx+256]; 
    __syncthreads();
  }  	       
  if(bdim>=256){
    if(tx<128) s_a[tx] += s_a[tx+128]; 
    __syncthreads();
  }
  if(bdim>=128){
    if(tx<64)  s_a[tx] += s_a[tx+64];
    __syncthreads();
  }
  if(tx<32){
    s_a[tx] += s_a[tx+32];
    s_a[tx] += s_a[tx+16]; 
    s_a[tx] += s_a[tx+8]; 
    s_a[tx] += s_a[tx+4];
    s_a[tx] += s_a[tx+2];
  }

  // final reduction
  if(tx==0)
    ra[bx] = sqrt((s_a[0] + s_a[1])/loops);
}
template <const unsigned int bdim, unsigned int loops, typename T, typename S>
__global__ void sumtime (S *a, T *ra){

  volatile __shared__  T s_a[bdim]; // force read from shared

  // assume a 1 dimensional thread array
  const unsigned int tx = threadIdx.x; // thread index in thread-block (0-indexed)
  const unsigned int bx = blockIdx.x;  // block index (0-indexed)

  // logical single index for this thread
  unsigned int n = tx;

  // is thread in range for the addition
  T d = 0.f;
  while(n<loops){
    d += (T)a[n+bx*loops];
    n += bdim;
  }
  
  // assume bx power of 2 
  s_a[tx] = d;
  __syncthreads();


  // unroll
  if(bdim>=1024){
    if(tx<512) s_a[tx] += s_a[tx+512]; 
    __syncthreads();
  }  	       
  if(bdim>=512){
    if(tx<256) s_a[tx] += s_a[tx+256]; 
    __syncthreads();
  }  	       
  if(bdim>=256){
    if(tx<128) s_a[tx] += s_a[tx+128]; 
    __syncthreads();
  }
  if(bdim>=128){
    if(tx<64)  s_a[tx] += s_a[tx+64];
    __syncthreads();
  }
  if(tx<32){
    s_a[tx] += s_a[tx+32];
    s_a[tx] += s_a[tx+16]; 
    s_a[tx] += s_a[tx+8]; 
    s_a[tx] += s_a[tx+4];
    s_a[tx] += s_a[tx+2];
  }

  // final reduction
  if(tx==0)
    ra[bx] = (s_a[0] + s_a[1])/loops;
}
bool __device__ update (float r1, float r2, float * prop,\
        unsigned int &pgal, unsigned int &mrna,\
        unsigned int &protein, float &time)
{
    bool go =0;
    float a0=0;
    float temp[7];//7 reactions
    temp[0]=(pgal+1)*prop[0];
    a0+=temp[0];
    temp[1]=prop[1];
    a0+=temp[1];
    temp[2]=pgal*prop[2];
    a0+=temp[2];
    temp[3]=prop[3];
    a0+=temp[3];
    temp[4]=mrna*prop[4];
    a0+=temp[4];
    temp[5]=mrna*prop[5];
    a0+=temp[5];
    temp[6]=protein*prop[6];
    a0+=temp[6];
    unsigned int id=10;
    float sum=0;
    float dt=(float)log(1/r1)/a0;
    a0*=r2;
    for (unsigned int i = 0; i < 6; i++) 
    {
        sum+=temp[i];
        if (a0<sum) {
            id=i;
            break;
        }
    }
    switch(id)
    {
        case 1: pgal++;
                break;
        case 2: mrna++;
                break;
        case 4: mrna--;
                break;
        case 5: protein++;
                go=1;
                break;
        case 6: protein--;
                go=1;
                break;
        default: break;
    }
    time+=dt;
    return go;
}

__device__ bool decay (float r1, float r2, float rate, unsigned int  &pgal,\
        unsigned int *mrna, unsigned int *protein, float &time, \
        float *time2, unsigned int selfid,bool factor)
{
    bool go=0;
    float a0=0;
    float temp[4];//7 reactions
    temp[0]=(pgal)*rate;
    a0+=temp[0];
    temp[1]=mrna[selfid]*rate;
    a0+=temp[1];
    temp[2]=protein[selfid]*rate;
    a0+=temp[2];
    unsigned int id=10;
    float sum=0;
    float dt=log(1/r1)/a0;
    a0*=r2;
    for (unsigned int i = 0; i < 3; i++) 
    {
        sum+=temp[i];
        if (a0<sum) {
            id=i;
            break;
        }
    }
    switch(id)
    {
        case 0: pgal+=factor==1?1:-1;
                break;
        case 1: mrna[selfid]+=factor==1?1:-1;
                break;
        case 2: protein[selfid]+=factor==1?1:-1;
                //if (target>0) protein[target]++;
                go=1;
        default: break;
    }
    time+=dt;
    return go;
    //if (target>0) time2[target]=time;
}
template <unsigned int offset >__device__ void writeout(float time, unsigned int p1, unsigned int p2, unsigned int *dev_out1,\
        unsigned int *dev_out2, float *dev_time, unsigned int counter)
{
    const int n=get_coord();
    dev_out1[counter+n*offset]=p1;
    dev_out2[counter+n*offset]=p2;
    dev_time[counter+n*offset]=time;
}
template  < const unsigned int loops, const unsigned int bdim, const unsigned int blocks  >\
void __global__ gillrun (unsigned int * seedpool, float * dev_prop, float cycletime, float starttime, \
        unsigned int * dev_out1, unsigned int *dev_out2, float * dev_time)
{
    int n= get_coord();
    unsigned int seed[6] ;
    const int tx = threadIdx.x; // thread index in thread-block (0-indexed)
    float ran[6];
    float c[9];
    for (int i = 0; i < 9; i++) {
        c[i]=dev_prop[i];//+9*blockIdx.x];
       // block index (0-indexed)
    }
    //update thread specific propensities
    __shared__ unsigned int pgal[bdim];
    __shared__ unsigned int mrna[bdim*2];
    __shared__ unsigned int protein[bdim*2];
    __shared__ float time[bdim];
    //volatile __shared__ int flip[bdim];
    pgal[tx]=50;
    mrna[tx]=0;
    mrna[tx+bdim]=0;
    protein[tx]=0;
    protein[tx+bdim]=0;
    time [tx]=0;
    //flip[tx]=0;

    //initialize shared memory;
    for (int i = 0; i < 6; i++) 
    {
        seed[i]=seedpool[n+i*bdim*blocks];
        __syncthreads();
    }
    //here we have loaded all the random seeds
    //if (tx==0)
    {
      //  flip[tx]=1;
      //  time[tx]=0;//start time
    }

    unsigned int counter=0;
    //bool over=0;
    float divide=c[8];
    //while (flip[tx]!=1){2*4.2+3.2;
    //if (flip[tx]==1) break;}; //hold threads not at time
    while (counter<loops&&time[tx]<starttime+cycletime*1e6) 
    {
        //we impose two jump out conditions
        lrg(seed);
        //one step randomize
        for (int i = 0; i < 6 ; i++) 
        {
            ran[i]=fr(seed, i);
        }
        //here we have one round of random number
        //set tnext
        //
        float tnext=time[tx];
        tnext+=cycletime*ran[3];
        //set tmiddle
        float tmiddle=time[tx]+cycletime*ran[5];
        //set tmiddle_2nd
        float tmiddle2=tmiddle+cycletime*ran[4];
        //for first part, while loop,  find the reaction number
        if (ran[2]>0.5) divide=1-divide;//randomize divide
        while (time[tx]<tmiddle)
        {
            bool go=0;
            if (counter>=loops) return;
            lrg(seed);
            for (int i = 0; i < 3; i++) {
                ran[i]=fr(seed,i);
            };            
            bool updated=(ran[2]>divide)?1:0;
            if(updated) go=update(ran[0],ran[1],c,pgal[tx],mrna[tx],protein[tx], time[tx]);
            else go=update(ran[0],ran[1],c,pgal[tx],mrna[tx+bdim],protein[tx+bdim], time[tx]);
            //update the first half reaction
            if (time[tx]>starttime&&go)
            {
                //writeout<loops>(seed[0],seed[1],seed[2], dev_out1, dev_out2,dev_time, counter);
                writeout<loops>(time[tx],protein[tx],protein[tx+bdim],dev_out1, dev_out2,dev_time, counter);
                counter++;
            }
        }
        while (time[tx]<tmiddle2)
        {
            bool go=0;
            if (counter>=loops) return;
            lrg(seed);
            for (int i = 0; i < 3; i++) {
                ran[i]=fr(seed,i);
            };         
            bool factor=ran[2]>0.5?1:0;
            if(fr(seed,0)>divide)
                go=decay(ran[0],ran[1],c[7], pgal[tx], mrna, protein, time[tx], time, tx, factor);
            else go=decay(ran[0],ran[1],c[7], pgal[tx], mrna, protein, time[tx], time, tx+bdim, factor);
            if (time[tx]>starttime&&go)
            {
                writeout<loops>(time[tx],protein[tx],protein[tx+bdim], dev_out1, dev_out2,dev_time, counter);
                counter++;
            }
        }
        //int target=-1;
        /* if(not over) for (unsigned int i = 0; i < bdim; i++) 
        {
            if (i==tx) continue;
            if (flip[i]==0)
            {
                target=i;
                flip[i]=-1;//in use
                break;
            }
            if (i==bdim-1) over=1;
        }
        */
        while (time[tx]<tnext)
        {
            bool go=0;
            if (counter>=loops) return;
            lrg(seed);
            for (int i = 0; i < 3; i++) {
                ran[i]=fr(seed,i);
            };         
            bool updated= (ran[2]>divide)?1:0;
            if(updated) go=decay(ran[0],ran[1],c[7], pgal[tx], mrna, protein, time[tx], time, tx, 0);
            else 
            {
                //if (target>0) target+=bdim;
                go=decay(ran[0],ran[1],c[7],pgal[tx], mrna, protein, time[tx],time, tx+bdim, 0);
            }
            //update the first half reaction
            if (time[tx]>starttime&&go)
            {
                writeout<loops>(time[tx],protein[tx],protein[tx+bdim], dev_out1, dev_out2,dev_time, counter);
                counter++;
            }
        }
        //if(target!=-1) flip[target]=1;//here start another cell
        //if(counter2>3)return;
    }

}

template <const unsigned int loops, const unsigned int offset >\
void __global__ init (float * dev_time, unsigned int * dev_out1, unsigned int * dev_out2)
{
    int n= get_coord();
    {
        for (int i=0; i < loops ; i++)
            {
                dev_time[i*offset + n]=0;
                dev_out1[i*offset + n]=0;
                dev_out2[i*offset + n]=0;
            }
    }
};


/*
 * =====================================================================================
 *        Class:  cudagill
 *  Description:  a class for a parallized simulator of a gillisy system with a primitive
 *                cell cycle like behavior
 *  Note:         only input coefficients that make sense
 * =====================================================================================
 */
class cudagill:public cudahelper
{
    public:
        /* ====================  LIFECYCLE     ======================================= */
        cudagill (float * prop, string sname) : cudahelper()
        {
            cellname=sname;
            cycletime=80;
            starttime=400;
            size_output=BDIM*LOOPS*BLOCKS*sizeof(unsigned int);
            size_time=BDIM*LOOPS*BLOCKS*sizeof(float);
            size_seedpool=BDIM*6*BLOCKS*sizeof(unsigned int);
            size_prop=BLOCKS*9*sizeof(float);
            size_avg=BLOCKS*BDIM*sizeof(float);
            cout << "sizes are " <<size_output <<" " <<size_time <<" "<<size_seedpool <<endl;
            //set size of arrays
            try 
            {
                host_seeds=new unsigned int [BDIM*6*BLOCKS];
                host_out=new unsigned int [BDIM*BLOCKS*LOOPS];
                host_out2=new unsigned int [BDIM*BLOCKS*LOOPS];
                host_time=new float [BDIM*BLOCKS*LOOPS];
                host_avg=new float [BDIM*BLOCKS];
                host_var=new float [BDIM*BLOCKS];
            }
            catch (...)
            {
                cout << "error assigning host memory"<<endl;
            }
            cout << "host memory assigned" <<endl;
            //assign predetermined propensities;
            dim3 temp (BLOCKS, 1,1);
            griddim=temp;
            dim3 temp2 (BDIM,1,1);
            blockdim = temp2;
            //assign propensities, here we assume same propensities for all BLOCKS
            try
            {
                if (cudaSuccess!= cudaMalloc(&dev_out1, size_output)) throw (1);
                if (cudaSuccess!= cudaMalloc(&dev_out2, size_output)) throw (2);
                if (cudaSuccess!= cudaMalloc(&dev_time, size_time)) throw (3);
                if (cudaSuccess!= cudaMalloc(&dev_seeds, size_seedpool)) throw (4);
                if (cudaSuccess!= cudaMalloc(&dev_prop, size_prop)) throw (5);

            }
            catch (cudaError_t err)
            {
                cout << "error assigning gpu mem" <<endl;
                checkerror(err);
            }
            cout << "gpu memory assigned" <<endl;
            plant();//plant the host seeds;

            setprop(prop);
            checkprop();
            cout << "checked prop" <<endl;


        };                             /* constructor */

        ~cudagill()
        {
            cout << "destructor called" <<endl;
            delete [] host_seeds, host_avg, host_var;
            delete [] host_out, host_out2;
            delete [] host_time;
            cout << "freeing device memories , numerous: " <<endl;
            checkerror(cudaFree(dev_prop));
            checkerror(cudaFree(dev_out1));
            checkerror(cudaFree(dev_out2));
            checkerror(cudaFree(dev_time));
            checkerror(cudaFree(dev_seeds));


        }
        /* ====================  ACCESSORS     ======================================= */
        void
            checkprop()
            {
                #if LOG
                cout << "checking copy back of propensities" <<endl;
                checkerror(cudaMemcpy(host_prop, dev_prop, size_prop, cudaMemcpyDeviceToHost));

                ofstream f2;
                f2.open("./Data/prop");

                for (int i = 0; i < BLOCKS; i++) {
                    for (int j=0; j<9; j++)
                        f2<<host_prop[i*9+j]<<" ";
                f2<<endl;
                }
                f2.close();
                #endif
            }
        void
            checkseed ( )
            {
                #if LOG
                cout << "checking copy back of seeds" <<endl;
                checkerror(cudaMemcpy(host_seeds, dev_seeds, size_seedpool, cudaMemcpyDeviceToHost));

                ofstream f2;
                f2.open("./Data/seeds");

                for (int i=0; i<BDIM*BLOCKS; i++)
                {
                    for (int j=0; j<6; j++)
                    {
                        f2<<host_seeds[i+j*BDIM*BLOCKS]<<" ";
                    }
                    f2<<endl;
                }
                f2.close();
                #endif
            }		/* -----  end of function checkseed  ----- */
        void
            writeresult()
            {
                #if 1
                ofstream f2;
                f2.open("./Data/probe.m");
                cout << "copying protein result 1 back " <<endl;
                checkerror(cudaMemcpy(host_out, dev_out1, size_output, cudaMemcpyDeviceToHost));

                for (int i=0; i<1;i++)//BDIM*BLOCKS; i++)
                {
                    f2<<"a=[";
                    for (int j=0; j<LOOPS; j++)
                    {
                        f2<<host_out[j+i*LOOPS]<<", ";
                    }
                    f2<<"];\n";
                }

                cout << "copying protein result 2 back " <<endl;
                checkerror(cudaMemcpy(host_out, dev_out2, size_output, cudaMemcpyDeviceToHost));


                for (int i=0; i<1;i++)//BDIM*BLOCKS; i++)
                {
                    f2<<"b=[";
                    for (int j=0; j<LOOPS; j++)
                    {
                        f2<<host_out[j+i*LOOPS]<<", ";
                    }
                    f2<<"];\n";
                }
                cout << "copying time result back " <<endl;
                checkerror(cudaMemcpy(host_time, dev_time, size_output, cudaMemcpyDeviceToHost));


                for (int i=0;i<1;i++)// i<BDIM*BLOCKS; i++)
                {
                    f2<<"t=[";
                    for (int j=0; j<LOOPS; j++)
                    {
                        f2<<host_time[j+i*LOOPS]<<", ";
                    }
                    f2<<"];\n";
                }
                f2<<"\nfigure\nplot(t,a,t,b,'LineWidth', 2);";
                f2.close();
                #endif
            }
        /* ====================  MUTATORS      ======================================= */
        void 
            setname (string name)
            {
                cellname=name;
            }
        void
            settime(float start2, float cycletime2)
            {
                starttime=start2;
                cycletime=cycletime2;

            }
        void
            setprop(float *prop)
            {
                //assigning kernel dimensions
                for (int i = 0; i < BLOCKS; i++) {
                    for (int j=0; j<9; j++)
                        host_prop[i*9+j]=prop[j];
                }
                try
                {
                    checkerror(cudaMemcpy(dev_prop, host_prop, size_prop, cudaMemcpyHostToDevice));
                }
                catch (...)

                {
                    cout << "error copying seeds to gpu" <<endl;
                    return;
                }
                cout << "host propensities copied" <<endl;
            }		/* -----  end of function plant  ----- */

        void
            plant (  )
            {
                srand(time(NULL));//initialize every time!
                for (int i =0; i < BDIM*6*BLOCKS; i++)
                    host_seeds[i]=(unsigned int)rand()+1;
                try
                {
                    checkerror(cudaMemcpy(dev_seeds, host_seeds, size_seedpool, cudaMemcpyHostToDevice));
                }
                catch (...)

                {
                    cout << "error copying seeds to gpu" <<endl;
                    return;
                }
                checkseed();
                cout << "checked seed" <<endl;
            }		/* -----  end of function plant  ----- */
        void
            setup()
            {
                cout << "initializing cuda arrays" <<endl;
                init <LOOPS, BDIM*BLOCKS> <<<griddim, blockdim>>> (dev_time, dev_out1, dev_out2);
                cudaError_t err= cudaGetLastError();
                checkerror(err);
            }

        /* ====================  OPERATORS     ======================================= */
        void
            execute()
            //execute kernels to get the raw data
            {
                plant();
                setup();
                Mark();
                cout << "now executing , wait... " <<endl;
                gillrun <LOOPS,BDIM,BLOCKS> <<<griddim, blockdim>>> (dev_seeds, dev_prop, cycletime, starttime,\
                        dev_out1, dev_out2, dev_time);
                cudaError_t err=cudaGetLastError();
                cout << "execution complete, checking error , elapsed time: " <<Elapsed()<<endl;
                cout << " conducting device wise sync" <<endl;
                checkerror(cudaDeviceSynchronize());
                checkerror(err);
                writeresult();
                maxval();
                sumele();
                stdsumele();
            }
        void
            writemat()
            {
                checkerror(cudaMemcpy(host_out, dev_out1, size_output, cudaMemcpyDeviceToHost));
                checkerror(cudaMemcpy(host_out2, dev_out2, size_output, cudaMemcpyDeviceToHost));
                unsigned int width=1000;//(unsigned int)host_max+1;
                unsigned int length=1000;//(unsigned int)host_max2+1;
                cout << "now writing png pics with dims: " <<host_max <<" " <<host_max2\
                    << " "<< width*length <<endl;
                float *count=new float[width*length];
                for (unsigned int i = 0; i < width*length; i++) 
                {
                    count[i]=0;
                }
                unsigned int cmp=0;
                for (unsigned int k = 0; k < BLOCKS; k++) 
                {
                    for (unsigned int i = 0; i < BDIM; i++) 
                    {
                        for (unsigned int j = 0; j < LOOPS; j++) 
                        {
                            cmp=cmp>host_out[j+i*LOOPS+k*BDIM*LOOPS]+host_out2[j+i*LOOPS+k*BDIM*BLOCKS]*width?\
                                cmp:host_out[j+i*LOOPS+k*BDIM*LOOPS]+host_out2[j+i*LOOPS+k*BDIM*BLOCKS]*width;
                            //cout << host_out[j+i*LOOPS]+host_out2[j+i*LOOPS]*width <<endl;
                            try
                            {
                                if (host_out[j+i*LOOPS+k*BDIM*LOOPS]>1000||\
                                        host_out2[j+i*LOOPS+k*BDIM*BLOCKS]>1000) continue;
                                count[host_out[j+i*LOOPS+k*BDIM*LOOPS]\
                                    +host_out2[j+i*LOOPS+k*BDIM*BLOCKS]*width]+=1;
                            }
                            catch (...)
                            {
                                cout << "failed count" <<endl;
                                break;
                            }
                        }
                    }
                }
                cout << "largest val is " <<cmp <<endl;
                #if LOG
                ofstream f2;
                f2.open("testingpng");
                for (unsigned int i = 0; i < width; i++) 
                {
                    for (unsigned int j = 0; j < length; j++) 
                    {
                        f2<<count[i+j*width]<<" ";
                    }
                    f2<<endl;
                }
                f2.close();
                #endif
                cout << "now printing png" <<endl;
                string fname="./PNG/"+cellname+".png";
                const char * fd=fname.c_str();
                WritePNG (fd, width, length, count);
                delete [] count;
            }
        void
            maxval()
            //get the max values of the total values;
            {
                const unsigned int interval=128;
                const unsigned int blocks2=(BLOCKS*BDIM*LOOPS+interval-1)/interval;
                //get enough large BLOCKS
                unsigned int *dev_temp;
                size_t sizetemp=BLOCKS*BDIM*LOOPS*sizeof(unsigned int);
                unsigned int *temp=new unsigned int[BLOCKS*BDIM*LOOPS];
                dim3 b1 (interval,1, 1);
                dim3 g1 (blocks2, 1,1);
                cout << "allocating cuda mem for maxval " <<endl;
                checkerror(cudaMalloc(&dev_temp, sizetemp));//we take an large enough output array
                getmax<interval, unsigned int> <<<g1,b1>>> (LOOPS*BDIM*BLOCKS, dev_out1, dev_temp);
                cout << "finished get max kernel for output1! "<<endl;
                checkerror (cudaGetLastError());
                checkerror(cudaDeviceSynchronize());
                cout << "now copying values back!" <<endl;
                checkerror(cudaMemcpy(temp, dev_temp, sizetemp, cudaMemcpyDeviceToHost));
                //copy up the values back
                unsigned int reda=0;
                for(int i=0;i<g1.x;++i)
                    {
                      reda = (reda>temp[i])?reda:temp[i];
                    }
                cout << "the largest value of output1 is " <<reda <<endl;
                for (unsigned int i = 0; i < BLOCKS*BDIM*LOOPS; i++) 
                {
                    temp[i]=0;
                }
                getmax<interval, unsigned int> <<<g1,b1>>> (LOOPS*BDIM*BLOCKS, dev_out2, dev_temp);
                cout << "finished get max kernel for output2! "<<endl;
                checkerror (cudaGetLastError());
                cout << "now copying values back!" <<endl;
                checkerror(cudaMemcpy(temp, dev_temp, sizetemp, cudaMemcpyDeviceToHost));
                //copy up the values back
                unsigned int reda2=0;
                for(int i=0;i<g1.x;++i)
                    {
                      reda2 = (reda2>temp[i])?reda2:temp[i];
                    }
                    cout << "the largest value of output2 is " <<reda2 <<endl;
                host_max=(float)reda;
                host_max2=(float) reda2;
                cout << "got the max value " <<host_max<<" "<<host_max2 <<endl;
                cudaFree(dev_temp);
                delete[] temp;
            }
        void
            stdsumele()
            {
                float *dev_temp;
                cudaMalloc(&dev_temp, size_output);
                const unsigned int interval=128;
                //100 threads per element-time with size loops
                const unsigned int blocks1=(BDIM*BLOCKS*LOOPS+interval-1)/interval;
                dim3 gd(blocks1, 1,1);
                dim3 bd(interval, 1, 1);
                varover<<<gd,bd>>>(BLOCKS*BDIM*LOOPS, dev_out1, dev_out2, dev_temp);
                checkerror(cudaDeviceSynchronize());
                const unsigned int blocks2=BDIM*BLOCKS;
                //each block responsible for one element
                dim3 gd2(blocks2, 1, 1);
                dim3 bd2(interval, 1, 1);
                cout << "attempting average , allocating gpu mem: " <<endl;
                checkerror (cudaMalloc(&dev_avg,size_avg));
                checkerror (cudaMalloc(&dev_var,size_avg));
                cout << "attempting average output 1: " <<endl;
                sumtime<interval, LOOPS, float, float> <<<gd2,bd2>>> \
                    (dev_temp, dev_avg);
                checkerror (cudaGetLastError());
                checkerror(cudaDeviceSynchronize());
                cout << "copying back average 1" <<endl;
                checkerror(cudaMemcpy(host_avg, dev_avg, size_avg, cudaMemcpyDeviceToHost));
                ofstream f1;
                string fname="./Data/rtoaverage_"+cellname+".m";
                f1.open(fname.c_str());
                f1<<"avg1=[";
                for (unsigned int i = 0; i < BLOCKS; i++) 
                {
                    for (unsigned int j = 0; j < BDIM; j++) 
                    {
                        f1<<host_avg[j+i*BDIM]<<" , ";
                    }
                }
                f1<<"];"<<endl;
                f1<<"avg=[avg1]';\n figure\nsubplot(2,2,1)\n \
                    hist(avg); \n subplot(2,2,2)\n probplot(avg);\n\
                    subplot (2,2,3); \n boxplot(avg);\n subplot(2,2,4)\n qqplot(avg);";
                f1.close();
                variance <interval, LOOPS, float,float><<<gd2,bd2>>>\
                    (dev_temp, dev_var, dev_avg );
                cout << "copying back variance 1" <<endl;
                checkerror(cudaMemcpy(host_var, dev_var, size_avg, cudaMemcpyDeviceToHost));
                fname=("./Data/rtovariance_"+cellname+".m");
                f1.open(fname.c_str());
                f1<<"var1=[";
                for (unsigned int i = 0; i < BLOCKS; i++) 
                {
                    //cout << "variance of element-block " <<i  <<" is " <<host_var[i] <<endl;
                    for (unsigned int j = 0; j < BDIM; j++) 
                    {
                        f1<<host_var[j+i*BDIM]<<" , ";
                    }
                }
                f1<<"];"<<endl;
                f1<<"var=[var1]';\n figure\nsubplot(2,2,1)\n \
                    hist(var); \n subplot(2,2,2)\n probplot(var);\n\
                    subplot (2,2,3); \n boxplot(var);\n subplot(2,2,4)\n qqplot(var)";
                f1.close();
                cudaFree(dev_var);
                cudaFree(dev_avg);
            }
        void
            sumele()
            //get the variance and average of each element
            {
                const unsigned int interval=128;
                //100 threads per element-time with size loops
                const unsigned int blocks2=BDIM*BLOCKS;
                //each block responsible for one element
                dim3 gd(blocks2, 1, 1);
                dim3 bd(interval, 1, 1);
                cout << "attempting average , allocating gpu mem: " <<endl;
                checkerror (cudaMalloc(&dev_avg,size_avg));
                checkerror (cudaMalloc(&dev_var,size_avg));
                cout << "attempting average output 1: " <<endl;
                sumtime<interval, LOOPS, float, unsigned int> <<<gd,bd>>> \
                    (dev_out1, dev_avg);
                checkerror (cudaGetLastError());
                cout << "copying back average 1" <<endl;
                checkerror(cudaMemcpy(host_avg, dev_avg, size_avg, cudaMemcpyDeviceToHost));
                ofstream f1;
                string fname="./Data/average_"+cellname+".m";
                f1.open(fname.c_str());
                f1<<"avg1=[";
                for (unsigned int i = 0; i < BLOCKS; i++) 
                {
                    for (unsigned int j = 0; j < BDIM; j++) 
                    {
                        f1<<host_avg[j+i*BDIM]<<" , ";
                    }
                }
                f1<<"];"<<endl;
                f1.close();
                variance <interval, LOOPS, float, unsigned int><<<gd,bd>>>\
                    (dev_out1, dev_var, dev_avg );
                cout << "copying back variance 1" <<endl;
                checkerror(cudaMemcpy(host_var, dev_var, size_avg, cudaMemcpyDeviceToHost));
                
                fname=("./Data/variance_"+cellname+".m");
                f1.open(fname.c_str());
                f1<<"var1=[";
                for (unsigned int i = 0; i < BLOCKS; i++) 
                {
                    //cout << "variance of element-block " <<i  <<" is " <<host_var[i] <<endl;
                    for (unsigned int j = 0; j < BDIM; j++) 
                    {
                        f1<<host_var[j+i*BDIM]<<" , ";
                    }
                }
                f1<<"];"<<endl;
                f1.close();
                cout << "attempting average output 2: " <<endl;
                sumtime<interval, LOOPS, float> <<<gd,bd>>> \
                    (dev_out2, dev_avg);
                checkerror (cudaGetLastError());
                cout << "copying back average 2" <<endl;
                checkerror(cudaMemcpy(host_avg, dev_avg, size_avg, cudaMemcpyDeviceToHost));

                
                fname="./Data/average_"+cellname+".m";
                f1.open(fname.c_str(),ios::app);
                f1<<"avg2=[";
                for (unsigned int i = 0; i < BLOCKS; i++) 
                {
                    for (unsigned int j = 0; j < BDIM; j++) 
                    {
                        f1<<host_avg[j+i*BDIM]<<" , ";
                    }
                }
                f1<<"];"<<endl;
                f1<<"avg=[avg1;avg2]';\n figure\nsubplot(2,2,1)\n \
                    hist(avg); \n subplot(2,2,2)\n probplot(avg);\n\
                    subplot (2,2,3); \n boxplot(avg);\n subplot(2,2,4)\n qqplot(avg);";
                f1.close();
                variance <interval, LOOPS, float><<<gd,bd>>>\
                    (dev_out2, dev_var, dev_avg );
                cout << "copying back variance 2" <<endl;
                checkerror(cudaMemcpy(host_var, dev_var, size_avg, cudaMemcpyDeviceToHost));
               
                fname=("./Data/variance_"+cellname+".m");
                f1.open(fname.c_str(),ios::app);
                f1<<"var2=[";
                for (unsigned int i = 0; i < BLOCKS; i++) 
                {
                    //cout << "variance of element-block " <<i  <<" is " <<host_var[i] <<endl;
                    for (unsigned int j = 0; j < BDIM; j++) 
                    {
                        f1<<host_var[j+i*BDIM]<<" , ";
                    }
                }
                f1<<"];"<<endl;
                f1<<"var=[var1;var2]';\n figure\nsubplot(2,2,1)\n \
                    hist(var); \n subplot(2,2,2)\n probplot(var);\n\
                    subplot (2,2,3); \n boxplot(var);\n subplot(2,2,4)\n qqplot(var)";
                f1.close();
                cudaFree(dev_var);
                cudaFree(dev_avg);
            }
    protected:
        dim3 griddim, blockdim;
        size_t size_seedpool, size_output, size_time, size_prop, size_avg;
        unsigned int *host_seeds, *dev_seeds, *host_out, *host_out2, *dev_out1, *dev_out2;
        float * dev_time, *host_time, host_prop[9*BLOCKS], \
            *dev_prop, *dev_max, *dev_avg, *dev_var,\
            host_max, host_max2, *host_avg, *host_var\
            ,cycletime, starttime;
        string cellname;
        //data container for yfp, mcherry data and time data;
        //note: does NOT use generic rtab and generic number of coefficients!
        /* ====================  DATA MEMBERS  ======================================= */

    private:
        /* ====================  DATA MEMBERS  ======================================= */

}; /* -----  end of class Cudagill  ----- */

