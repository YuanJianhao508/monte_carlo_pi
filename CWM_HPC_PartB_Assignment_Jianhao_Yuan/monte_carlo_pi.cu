/*CWM HPC Part B Assignment: Monte Carlo Method for calculate pi value on GPU
2021/5/58 Jianhao Yuan */
// reference: https://blog.csdn.net/ichocolatekapa/article/details/18960223

//import libs
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//curand for random points generate
#include <curand.h>
#include <curand_kernel.h>

//Define constants (use 256 threads, and max trial times: 2000000)
#define MAX_THREAD  256
#define MAX_COUNT   2000000

//Kernel
__global__ void get_pi(float *res,int *count, int *time){
   //declare variables: 
   //initial # points in 1/4 circle; total number of random point generated: n;
   //time for running time test: t; loop index:i 
   int a=0, index_x = threadIdx.x, n = *count, t = *time, i;

   // declare coordinate variables x,y
   float x, y;

   // result for pi record
   res += index_x;

   //use curand to get random points
   curandState s;
   curand_init(t, index_x, 0, &s);
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

    //count/time (both host&device);loop index needed
    int maxThread = MAX_THREAD, *h_count, *d_count, *h_time, *d_time, i;

    //allocate memory for host
    h_pi = (float *)malloc(sizeof(float) * maxThread);
    h_count = (int *)malloc(sizeof(int) * 1);
    h_time = (int *)malloc(sizeof(int) * 1);

    //allocate memory for device
   cudaMalloc((void **)&d_pi, sizeof(float) * maxThread);
   cudaMalloc((void **)&d_count, sizeof(int) * 1);
   cudaMalloc((void **)&d_time, sizeof(int) * 1);

   //initialize count/time on host
   h_count[0] = MAX_COUNT;
   h_time[0] = (int)time(NULL);

   //get count&time value to device
   cudaMemcpy(d_count, h_count, sizeof(int) * 1, cudaMemcpyHostToDevice);
   cudaMemcpy(d_time, h_time, sizeof(int) * 1, cudaMemcpyHostToDevice);

   //running time test:start
   clock_t start, end;
   start = clock();

   //execute kernel
   get_pi<<<1, maxThread>>> (d_pi, d_count, d_time);

   //get pi value back to host
   cudaMemcpy(h_pi, d_pi, sizeof(float) * maxThread,cudaMemcpyDeviceToHost);

   //average over 512 threads
   for (i = 0; i < maxThread; i++) pi += h_pi[i];
   pi = pi / maxThread;

   //running time test:start
   end = clock();

   //Find error
   err = pi - (float)M_PI;
   if (err < 0) {
       err = -err;
    }

   //print output
   printf("Points: %d, Generated π: %f, Error: %.0fe-6\n",h_count[0] * maxThread, pi, err * 1000000);
   printf("Timer: %f sec\n", (float)(end -start)/CLOCKS_PER_SEC);

   //free memory on host
   free(h_pi);
   free(h_count);
   free(h_time);

   //free memory on device
   cudaFree(d_pi);
   cudaFree(d_count);
   cudaFree(d_time);

   //end
   return 0;
}
