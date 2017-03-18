#ifndef UTILS_H
#define UTILS_H

#include "common.h"

__host__ __device__ inline int mini2(int a, int b)
{
	return a < b ? a : b;
}

__host__ __device__ inline int maxi2(int a, int b)
{
	return a > b ? a : b;
}

__host__ __device__ inline float minf2(float a, float b)
{
	return a < b ? a : b;
}
__host__ __device__ inline float minf3(float a, float b, float c)
{
	float m = minf2(b, c);
	return a < m ? a : m;
}
__host__ __device__ inline float minf4(float a, float b, float c, float d)
{
	float m = minf3(b, c, d);
	return a < m ? a : m;
}

__host__ __device__ inline float maxf2(float a, float b)
{
	return a > b ? a : b;
}
__host__ __device__ inline float maxf3(float a, float b, float c)
{
	float m = maxf2(b, c);
	return a > m ? a : m;
}
__host__ __device__ inline float maxf4(float a, float b, float c, float d)
{
	float m = maxf3(b, c, d);
	return a > m ? a : m;
}

#endif // UTILS_H
