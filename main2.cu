/*
   here are the basic ideas:
   1. create a host class :: cudahelper that has a kernel manager
   2. intiialize by specifying the propensities and the initial values for each block. plus the divide value.
   3. each block start with BDIM threads that grab the initializers accordingly
   4. implement a cuda rannum gen
   5. each block create BDIM large shared arrays of initializers arrays for each values to be passed to the next cell
   6. there are also BDIM*GDIM*NLOOPS large arrays of two protein arrays and time arrays global to record the raw out data
   7. inside the thread there are latest values to run the update function
   8. device fctn to update and decide which reaction to run (return reaction number)
   9. thread has a switch to update that local reaction and write to global. no global read in all this.
   10. at each round if not initialized check shared memory for clue to init.
   11. to divide, loop through share memory bool flip array and write to the latest.
   
   12. when done reduce the global arrays to obtain statistical values.

   */

#include "cudagill.hpp"

int main (int argc, char **argv)
{
    float c[9]= { 1, 0.75 ,0.05, 0.0025 , 0.05, 0.1, 0.001, 0.025, 0.5};
    float c2[9]= { 1, 0.75 ,0.05, 0.0025 , 0.05, 0.1, 0.001, 0.025, 0.7};
    float c3[9]= { 1, 0.75 ,0.05, 0.0025 , 0.05, 0.1, 0.001, 0.025, 0.8};
    float c4[9]= { 1, 0.75 ,0.05, 0.0025 , 0.05, 0.1, 0.001, 0.025, 0.9};
    
    cudagill gill1(c,"div50");
    gill1.execute();
    gill1.writemat();
    gill1.setprop(c2);
    gill1.setname("div70");
    gill1.execute();
    gill1.writemat();
    gill1.setprop(c3);
    gill1.setname("div80");
    gill1.execute();
    gill1.writemat();
    gill1.setprop(c4);
    gill1.setname("div90");
    gill1.execute();
    gill1.writemat();
/*
    checkerror(cudaMemcpy(host_Ran, dev_Ran, size, cudaMemcpyDeviceToHost));
    ofstream f1;
    f1.open ("data.txt");
    for  (int i=0; i <LOOPS; i++)
    {
        for (int j=0; j<BDIM*BLOCKS; j++)
        {
            f1<<host_Ran[j+i*BDIM*BLOCKS]<<" ";
        }
        f1<<endl;
        cout <<"loop "<<i<<" complete "<<endl;
    }
    f1.close();
*/
}
