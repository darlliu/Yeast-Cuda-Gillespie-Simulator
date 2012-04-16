// utility class for cuda computations

typedef enum {
  inuse = 1,
  avail = 2
}cumemstatus;

class cumem;
static cumem *cumembase = NULL;

class cumem{
private:
  int sz;
  void *data;
  cumemstatus status;
  cumem *next;
public:
  
  cumem(int newsz){
    sz = newsz;
    cudaMalloc(&data, sz);
    status = inuse;
    next = NULL;
  };
  
  void *find(int newsz){
    
    for(cumem *cm = cumembase;cm;cm=cm->next){
      if(cm->sz==newsz && cm->status==avail){
	// reuse existing array
	cm->status = inuse;
	return cm->data;
      }
      else if(cm->next==NULL){
	// create new array
	cm->next = new cumem(newsz);
	return cm->next->data;
      }
    }
    
    // should throw an exception
    return NULL;
  }
  
  void recycle(void *pt){
    for(cumem *cm = cumembase;cm;cm=cm->next){
      if(cm->data==pt){
	
	// return array to available status
	  cm->status = avail;
      }
    }
  }

  void report(){
    for(cumem *cm = cumembase;cm;cm=cm->next){
      cout << "cmem ----- " << endl;
      cout << "sz: " << cm->sz << " data: " << cm->data;
      cout << " status: " << cm->status << " next: " << cm->next;
      cout << endl;
    }
  }
};

// helper class for cuda operations
class cudahelper{
  
private:

  int devnum;

  cudaEvent_t start;
  cudaEvent_t stop;

  int status;

public:

  cudahelper(){
    status = 0;

    // find the number of the device being used
    cudaGetDevice(&devnum);
  }

  void SetDevice(int dev){
    
    // set the device being used
    devnum = dev;
    cudaSetDevice(devnum);
  }

  void Mark(){
    // insert a "start" event into the command stream
    cudaEventCreate(&start); 

    // instert a "stop" event into the command stream
    cudaEventCreate(&stop);    

    // record the event
    cudaEventRecord( start, 0 );
    status = 1;
  }

  // elapsed time since mark was called in seconds
  float Elapsed(){

    float ftime = 0;

    if(status==1){
      // mark the stop event
      cudaEventRecord(stop, 0); 
      cudaEventSynchronize( stop );    
      cudaEventElapsedTime( &ftime, start, stop ); 

      // tidy up
      cudaEventDestroy( start ); 
      cudaEventDestroy( stop );
      status = 0;

      // scale for milliseconds
      ftime *= 1e-3;

    }

    return ftime;
  }

  void CheckError(const char *msg){

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
      { 
	std::cout << "Cuda error: " << msg
		  << ": " << cudaGetErrorString(err) << std::endl;
      }         
  }
  
  void BufferedMalloc(void **pt, int sz){
    if(!cumembase){
      cumembase = new cumem(sz);
    }
    *pt = cumembase->find(sz);
  }
  
  void BufferedFree(void *pt){
    if(!cumembase){
      // should throw error
    }
    cumembase->recycle(pt);
  }

  void BufferedReport(){
    cumembase->report();
  }
  


};




