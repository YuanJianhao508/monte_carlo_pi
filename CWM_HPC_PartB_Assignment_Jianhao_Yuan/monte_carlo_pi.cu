/*CWM HPC Part B Assignment: Monte Carlo Method for calculating pi value on GPU
2021/5/58 Jianhao Yuan */
// reference: https://blog.csdn.net/ichocolatekapa/article/details/18960223

//import libs
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
//curand for random points generate
#include <curand.h>
#include <curand_kernel.h>

//Define constants (use 256 threads, and max trial times: 2000000)
#define MAX_THREAD  256
#define MAX_COUNT   2000000

//Kernel
__global__ void get_pi(float *res,int *count){
   //declare variables: 
   //initial # points in 1/4 circle; total number of random point generated: n; loop index:i 
   int a=0, index_x = threadIdx.x, n = *count,i;

   // declare coordinate variables x,y
   float x, y;

   // result for pi record
   res += index_x;

   //use curand to get random points
   curandState s;
   curand_init(42, index_x, 0, &s);
   for (i = 1; i <= n; i++) {
       //random generate in 1*1 square
       x = curand_uniform(&s);
       y = curand_uniform(&s);
       //count in if point locate in 1/4 circle
       if (pow(x, 2) + pow(y, 2) <= 1) {
           a++;
       }
       //get pi value
       *res = 4 * (float)a / (float)n;

       //synchronzie threads
       __syncthreads();
    }
}


int main(void){
    // declare variables: host pi value, device pi value, actual pi value, error between
    float *h_pi, *d_pi, pi, err;

    //count(both host&device);loop index needed
    int maxThread = MAX_THREAD, *h_count, *d_count, i;

    //allocate memory for host
    h_pi = (float *)malloc(sizeof(float) * maxThread);
    h_count = (int *)malloc(sizeof(int) * 1);

    //allocate memory for device
   cudaMalloc((void **)&d_pi, sizeof(float) * maxThread);
   cudaMalloc((void **)&d_count, sizeof(int) * 1);


   //initialize count number on host
   h_count[0] = MAX_COUNT;


   //get count value to device
   cudaMemcpy(d_count, h_count, sizeof(int) * 1, cudaMemcpyHostToDevice);

   //execute kernel
   get_pi<<<1, maxThread>>> (d_pi, d_count);

   //get pi value back to host
   cudaMemcpy(h_pi, d_pi, sizeof(float) * maxThread,cudaMemcpyDeviceToHost);

   //average over 512 threads
   for (i = 0; i < maxThread; i++) pi += h_pi[i];
   pi = pi / maxThread;

   //Find error
   err = pi - (float)M_PI;
   if (err < 0) {
       err = -err;
    }

   //print output
   printf("Points: %d, Generated π: %f, Error: %.0fe-6\n",h_count[0] * maxThread, pi, err * 1000000);

   //free memory on host
   free(h_pi);
   free(h_count);

   //free memory on device
   cudaFree(d_pi);
   cudaFree(d_count);

   //end
   return 0;
}