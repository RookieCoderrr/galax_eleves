#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__device__ float3 sub(float3 &a, float3 &b) {

  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

__device__ float3 add(float3 &a, float3 &b) {

  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ float3 multi_1(float3 &a, float3 &b) {

  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);

}

__device__ float3 multi_2(float3 &a, float b) {

  return make_float3(a.x * b, a.y * , a.z * b);

}


__device__ float3 multi_3(float3 &a, float b, float c) {

  return make_float3(a.x * b * c, a.y * b * c, a.z * b * c);

}

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j = 0; j < n_particles; j++)
	{
		if(i != j)
		{
			const float3 diff = sub(positionsGPU[j] , positionsGPU[i]);
		
			float3  res = multi_1(diff, diff);
			float dij = res.x + res.y + res.z;
			
			if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = rsqrtf(dij);
				dij = 10.0 * (dij * dij * dij);
			}
			float3 acc = multi_3(diff, dij, massesGPU[j]);
			accelerationsGPU[i] = add(accelerationsGPU[i], acc);
			
		}
	}
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	velocitiesGPU[i] = add(velocitiesGPU[i], multi_2(accelerationsGPU[i], 2.0f));
	positionsGPU[i] = add(positionsGPU[i], multi_2(velocitiesGPU[i], 0.1f));

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU);
}


#endif // GALAX_MODEL_GPU