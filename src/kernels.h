#ifndef CPU_H
#define CPU_H

#include "types.h"

#define BLOCK_X    8
#define BLOCK_Y    8

#define BLOCK      64
#define BLOCKS_ROW 64

class Timer
{
	clock_t _start, _end;
public:
	void start()       { _start = clock(); }
	void stop()        { _end   = clock(); }
	float time() const { return (float) (_end - _start) / CLOCKS_PER_SEC; }
};

#if CPU
	typedef Timer CudaTimer;
#else
	class CudaTimer
	{
		cudaEvent_t _start, _end;
	public:
		CudaTimer()
		{
			CUDA_SAFE_CALL(cudaEventCreate(&_start));
			CUDA_SAFE_CALL(cudaEventCreate(&_end));
		}
		void start()
		{
			CUDA_SAFE_CALL(cudaEventRecord(_start));
			CUDA_SAFE_CALL(cudaEventSynchronize(_start));
		}
		void stop()
		{
			CUDA_SAFE_CALL(cudaEventRecord(_end));
			CUDA_SAFE_CALL(cudaEventSynchronize(_end));
		}
		float time() const
		{
			float time;
			CUDA_SAFE_CALL(cudaEventElapsedTime(&time, _start, _end));
			return time / 1000.0f;
		}
	};
#endif

#define TEST    0
#define CLOSEST 1

float4 *___tris;
int    *___grid;

#if CPU
	inline __host__   float4 vertex(int i) { return ___tris[i]; }
	inline __host__   int    grid(int i)   { return ___grid[i]; }
#else
	texture<float4, 1> texTris;
	texture<int,    1> texGrid;
	inline __device__ float4 vertex(int i) { return tex1Dfetch(texTris, i); }
	inline __device__ int    grid(int i)   { return tex1Dfetch(texGrid, i); }
#endif

#if !USE_GRID

	template<int type> __device__ inline int rayScene
	(
		const float3& min,  const float3& max, const int3& div,
		const float3& orig, const float3& dir, float tMax,
		int trisNum,
		float *tResult
	)
	{
		int result = (type == TEST) ? 0 : -1;
		float tMin = 10e9;
		float t;

		for (int i = 0; /* at the end */; ++i)
		{
			float4 v0 = vertex(i * 3);
			float4 e1 = vertex(i * 3 + 1);
			float4 e2 = vertex(i * 3 + 2);
		
			if (type == TEST)
			{
				if (rayTriTest(orig, dir, v0, e1, e2, &t) && t > 0.0001f && t < tMax)
				{
					result = 1;
				}
			}
			else // CLOSEST
			{
				if (rayTriTest(orig, dir, v0, e1, e2, &t) && t > -0.0001f && t < tMin)
				{
					tMin = t;
					result = i;
				}
			}
		
			if (i >= trisNum)
			{
				break;
			}
		}
	
		if (type == CLOSEST)
		{
			*tResult = tMin;
		}
		return result;
	}

#else // USE_GRID

	template<int type> __device__ inline int rayCell
	(
		const float3& orig, const float3& dir, float tMax,
		int n, int offset,
		float *tResult
	)
	{
		int result = (type == TEST) ? 0 : -1;
		float tMin = 10e9;
		float t;

		for (int i = offset; ; ++i)
		{
			float4 v0 = vertex(i * 3);
			float4 e1 = vertex(i * 3 + 1);
			float4 e2 = vertex(i * 3 + 2);

			if (type == TEST)
			{
				if (rayTriTest(orig, dir, v0, e1, e2, &t) && t > 0.01f && t < tMax)
				{
					result = 1;
				}
			}
			else // CLOSEST
			{
				if (rayTriTest(orig, dir, v0, e1, e2, &t) && t > -0.01f && t < tMin)
				{
					tMin = t;
					result = i;
				}
			}
		
			if (i >= offset + n - 1)
			{
				break;
			}
		}

		if (type == CLOSEST)
		{
			*tResult = tMin;
		}
		return result;
	}

	template<int type> __device__ inline int rayScene
	(
		const float3& min,  const float3& max, const int3& div,
		const float3& orig, const float3& dir, float tMax,
		int trisNum,
		float *tResult
	)
	{
		float tMin;
		if (!rayBoxTest(orig, dir, min, max, &tMin))
		{
			return (type == TEST) ? 0 : -1;
		}
		if (tMin < 0) tMin = 0; // !important

		float3 start = orig + dir * tMin;
		float3 size  = max - min;
		float3 voxel = make_float3(size.x / div.x, size.y / div.y, size.z / div.z);
		
		int x = (int) ((start.x - min.x) / size.x * div.x);
		int y = (int) ((start.y - min.y) / size.y * div.y);
		int z = (int) ((start.z - min.z) / size.z * div.z);
		x -= (x == div.x);
		y -= (y == div.y);
		z -= (z == div.z);

		int stepX = 2 * (dir.x >= 0) - 1;
		int stepY = 2 * (dir.y >= 0) - 1;
		int stepZ = 2 * (dir.z >= 0) - 1;
		
		float tVoxelX = (float) (x + (dir.x >= 0)) / div.x;
		float tVoxelY = (float) (y + (dir.y >= 0)) / div.y;
		float tVoxelZ = (float) (z + (dir.z >= 0)) / div.z;

		float voxelMaxX = min.x + tVoxelX * size.x;
		float voxelMaxY = min.y + tVoxelY * size.y;
		float voxelMaxZ = min.z + tVoxelZ * size.z;

		float tMaxX = tMin + (voxelMaxX - start.x) / dir.x;
		float tMaxY = tMin + (voxelMaxY - start.y) / dir.y;
		float tMaxZ = tMin + (voxelMaxZ - start.z) / dir.z;

		float tDeltaX = voxel.x / fabs(dir.x);
		float tDeltaY = voxel.y / fabs(dir.y);
		float tDeltaZ = voxel.z / fabs(dir.z);
		
		for (;;)
		{
			int offset = 3 + div.x * div.y * z + div.x * y + x;
			int nodeOffset = grid(offset);
			
			float min = tMaxX;
			if (min > tMaxY) min = tMaxY;
			if (min > tMaxZ) min = tMaxZ;
		
			if (nodeOffset == -1) // empty cell
			{
				goto NEXT_VOXEL;
			}
		
			// cell
			{
				int trisNum = -grid(nodeOffset);
				int trisOffset = grid(nodeOffset + 1);
				
				if (type == TEST)
				{
					if (rayCell<TEST>(orig, dir, tMax, trisNum, trisOffset, 0))
					{
						return 1;
					}
				}
				else // type == CLOSEST
				{
					int triangleIndex = rayCell<CLOSEST>(orig, dir, 0, trisNum, trisOffset, tResult);
					if (triangleIndex != -1 && *tResult < min)
					{
						return triangleIndex;
					}
				}
			}
		
			NEXT_VOXEL:
			
			// ray went too far
			if (type == TEST && min > tMax)
			{
				break;
			}

			// simple Woo
			if (tMaxX <= tMaxY && tMaxX <= tMaxZ)
			{
				x += stepX;
				tMaxX += tDeltaX;
			}
			else if (tMaxY <= tMaxZ && tMaxY <= tMaxX)
			{
				y += stepY;
				tMaxY += tDeltaY;
			}
			else
			{
				z += stepZ;
				tMaxZ += tDeltaZ;
			}
		
			// went outside grid
			if (x < 0 || x >= div.x || y < 0 || y >= div.y || z < 0 || z >= div.z)
			{
				break;
			}
		}
	
		return (type == TEST) ? 0 : -1;
	}
	
	__device__ inline int shortTest
	(
		const float3& min,  const float3& max, const int3& div,
		const float3& from, const int3& from_cell,
		const float3& to,
		int trisNum
	)
	{
		int x = from_cell.x;
		int y = from_cell.y;
		int z = from_cell.z;
		
		float3 dir  = to - from;
		float tMax  = len(dir);
		dir /= tMax;
		
		float3 size  = max - min;
		float3 voxel = make_float3(size.x / div.x, size.y / div.y, size.z / div.z);
	
		int stepX = 2 * (dir.x >= 0) - 1;
		int stepY = 2 * (dir.y >= 0) - 1;
		int stepZ = 2 * (dir.z >= 0) - 1;

		float tVoxelX = (float) (x + (dir.x >= 0)) / div.x;
		float tVoxelY = (float) (y + (dir.y >= 0)) / div.y;
		float tVoxelZ = (float) (z + (dir.z >= 0)) / div.z;

		float voxelMaxX = min.x + tVoxelX * size.x;
		float voxelMaxY = min.y + tVoxelY * size.y;
		float voxelMaxZ = min.z + tVoxelZ * size.z;

		float tMaxX = (voxelMaxX - from.x) / dir.x;
		float tMaxY = (voxelMaxY - from.y) / dir.y;
		float tMaxZ = (voxelMaxZ - from.z) / dir.z;

		float tDeltaX = voxel.x / fabs(dir.x);
		float tDeltaY = voxel.y / fabs(dir.y);
		float tDeltaZ = voxel.z / fabs(dir.z);
	
		for (;;)
		{
			int offset = 3 + div.x * div.y * z + div.x * y + x;
			int nodeOffset = grid(offset);
			
			float min = tMaxX;
			if (min > tMaxY) min = tMaxY;
			if (min > tMaxZ) min = tMaxZ;
		
			if (nodeOffset == -1) // empty cell
			{
				goto NEXT_VOXEL;
			}
		
			// cell
			{
				int trisNum = -grid(nodeOffset);
				int trisOffset = grid(nodeOffset + 1);
			
				if (rayCell<TEST>(from, dir, tMax, trisNum, trisOffset, 0))
				{
					return 1;
				}
			}
		
			NEXT_VOXEL:
			
			// stop right there!
			if (min > tMax)
			{
				break;
			}

			// simple Woo
			if (tMaxX <= tMaxY && tMaxX <= tMaxZ)
			{
				x += stepX;
				tMaxX += tDeltaX;
			}
			else if (tMaxY <= tMaxZ && tMaxY <= tMaxX)
			{
				y += stepY;
				tMaxY += tDeltaY;
			}
			else
			{
				z += stepZ;
				tMaxZ += tDeltaZ;
			}
			
			// went outside grid
			if (x < 0 || x >= div.x || y < 0 || y >= div.y || z < 0 || z >= div.z)
			{
				break;
			}
		}
	
		return 0;
	}

#endif

__device__ inline int3 cell(const float3& v, const float3& min, const float3& size, const int3& div)
{
	int3 r;
	r.x = (int) ((v.x - min.x) / size.x * div.x);
	r.y = (int) ((v.y - min.y) / size.y * div.y);
	r.z = (int) ((v.z - min.z) / size.z * div.z);
	r.x -= (r.x == div.x);
	r.y -= (r.y == div.y);
	r.z -= (r.z == div.z);
	return r;
}

/*
 * CAST ONE PRIMARY RAY
 */
__global__ void castPrimary
(
	int width, int height,
#if CPU
	int x, int y,
#endif
	float3 min, float3 max, int3 div,
	float3 camPos, float3 camDir, float3 hStep, float3 vStep,
	int trisNum,
	float4 *hits
)
{
#if GPU
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
#endif
	
	float3 dir = camDir + hStep * (x - width  / 2) + vStep * (y - height / 2);
	normalize(dir);
	float t;
	int triangleIndex = rayScene<CLOSEST>(min, max, div, camPos, dir, 0, trisNum, &t);
	float3 hit = camPos + dir * t;
	hits[y * width + x] = make_float4(hit.x, hit.y, hit.z, triangleIndex);
}

/*
 * CAST PRIMARY RAYS
 */
__host__ float4 *castPrimaryRays
(
	const Scene& s
)
{
	// calculate screen rays' offsets
	float3 right = cross(s.camDir, s.camUp);
	normalize(right);
	float w = tan(s.hFov / 2.0f) * 2.0f;
	float h = tan(s.vFov / 2.0f) * 2.0f;
	float3 hStep = right * (w / 2) / (s.width  / 2);
	float3 vStep = -s.camUp * (h / 2) / (s.height / 2);
	
	#if CPU

		float4 *hits = new float4[s.width * s.height];

		Timer timer;
		timer.start();
		for (int y = 0; y < s.height; ++y)
		{
			for (int x = 0; x < s.width; ++x)
			{
				castPrimary(s.width, s.height, x, y, s.min, s.max, s.div, s.camPos, s.camDir, hStep, vStep, s.trisNum, hits);
			}
			if (y % 10 == 0)
			{
				printf("\rPrimary rays: %d%%", (int) ((float) y / s.height * 100));
				fflush(stdout);
			}
		}
		timer.stop();
		printf("\rPrimary rays: %.2f sec\n\n", timer.time());
	
		return hits;
	
	#else // GPU

		float4 *_hits;
		CUDA_SAFE_CALL(cudaMalloc(&_hits, sizeof(float4) * s.width * s.height));
	
		int block_x = BLOCK_X;
		int block_y = BLOCK_Y;
		int blocks_x = s.width  / block_x + (s.width  % block_x != 0);
		int blocks_y = s.height / block_y + (s.height % block_y != 0);

		CudaTimer timer;
		timer.start();
		castPrimary<<<dim3(blocks_x, blocks_y), dim3(block_x, block_y)>>>(s.width, s.height, s.min, s.max, s.div, s.camPos, s.camDir, hStep, vStep, s.trisNum, _hits);
		CUDA_SAFE_CALL(cudaGetLastError());
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		timer.stop();
		printf("\rPrimary rays: %.2f sec\n\n", timer.time());
	
		return _hits;
	
	#endif
}

/*
 * CAST ONE SHADOW RAY (SHARP SHADOWS)
 */
__global__ void castShadow
(
	const float4 *hits,
	int width, int height,
#if CPU
	int x, int y,
#endif
	float3 min, float3 max, int3 div,
	float3 camPos,
	float3 lightPos,
	int trisNum,
	int *visible
)
{
#if GPU
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
#endif
	
	float4 hit = hits[width * y + x];
	if (hit.w < 0)
	{
		return;
	}
	
	int triangleIndex = roundf(hit.w);
	float4 e1 = vertex(triangleIndex * 3 + 1);
	float4 e2 = vertex(triangleIndex * 3 + 2);
	float3 n;
	CROSS(n, e1, e2);
	if (DOT(n, camPos - hit) < 0) // normal should look at us
	{
		n = -n;
	}
	
	float3 dir = lightPos - hit;
	if (DOT(n, dir) < 0) // on the dark side
	{
		visible[y * width + x] = 0;
		return;
	}
	
	float dist = len(dir);
	dir /= dist;
	
	visible[y * width + x] = !rayScene<TEST>(min, max, div, make_float3(hit.x, hit.y, hit.z), dir, dist, trisNum, NULL);
}

/*
 * CAST SHADOW RAYS (SHARP SHADOWS)
 */
__host__ void castShadowRays
(
	const float4 *hits,
	const Scene& s,
	const float3& lightPos,
	int *visible
)
{
	#if CPU

		for (int y = 0; y < s.height; ++y)
		{
			for (int x = 0; x < s.width; ++x)
			{
				castShadow(hits, s.width, s.height, x, y, s.min, s.max, s.div, s.camPos, lightPos, s.trisNum, visible);
			}
		}
	
	#else // GPU

		int block_x = BLOCK_X;
		int block_y = BLOCK_Y;
		int blocks_x = s.width  / block_x + (s.width  % block_x != 0);
		int blocks_y = s.height / block_y + (s.height % block_y != 0);
	
		castShadow<<<dim3(blocks_x, blocks_y), dim3(block_x, block_y)>>>(hits, s.width, s.height, s.min, s.max, s.div, s.camPos, lightPos, s.trisNum, visible);
		CUDA_SAFE_CALL(cudaGetLastError());

	#endif
}

__host__ __device__ inline int3 nearest(const float4& v, const float3& min, const float3& size, const int3& div)
{
	int3 r;
	r.x = (int) ((v.x - min.x) / size.x * div.x + 0.5f);
	r.y = (int) ((v.y - min.y) / size.y * div.y + 0.5f);
	r.z = (int) ((v.z - min.z) / size.z * div.z + 0.5f);
	return r;
}
__host__ __device__ inline int3 cell(const float4& v, const float3& min, const float3& size, const int3& div)
{
	int3 r;
	r.x = (int) ((v.x - min.x) / size.x * div.x);
	r.y = (int) ((v.y - min.y) / size.y * div.y);
	r.z = (int) ((v.z - min.z) / size.z * div.z);
	r.x -= (r.x == div.x);
	r.y -= (r.y == div.y);
	r.z -= (r.z == div.z);
	return r;
}
__host__ __device__ inline int index(const int3& v, const int3& div)
{
	return div.y * div.x * v.z + div.x * v.y + v.x;
}
__host__ __device__ inline bool in_bounds(const int3& v, const int3& div)
{
	return v.x >= 0 && v.x < div.x
	    && v.y >= 0 && v.y < div.y
		&& v.z >= 0 && v.z < div.z;
}

__host__ __device__ inline void set_bit(bits256& num, unsigned off)
{
	unsigned char *bytes = (unsigned char *) &num;
	unsigned byte = off >> 3;
	unsigned byte_off = off & 7;
	bytes[byte] |= 1 << byte_off;
}

__host__ __device__ inline int get_bit(const bits256& num, unsigned off)
{
	const unsigned char *bytes = (const unsigned char *) &num;
	unsigned byte = off >> 3;
	unsigned byte_off = off & 7;
	return bytes[byte] & (1 << byte_off);
}

__host__ __device__ inline void set_bit(int& num, unsigned off)
{
	unsigned char *bytes = (unsigned char *) &num;
	unsigned byte = off >> 3;
	unsigned byte_off = off & 7;
	bytes[byte] |= 1 << byte_off;
}

__host__ __device__ inline int get_bit(const int& num, unsigned off)
{
	const unsigned char *bytes = (const unsigned char *) &num;
	unsigned byte = off >> 3;
	unsigned byte_off = off & 7;
	return bytes[byte] & (1 << byte_off);
}

__global__ void shortTest
(
	const float4 *hits,
	int width, int height,
#if CPU
	int x, int y,
#endif
	float3 min, float3 max, int3 div, int3 lm_div,
	float3 camPos,
	const int3 *tpl, int tpl_size, float radius,
	int trisNum,
	bits256 *tpl_used
)
{
#if GPU
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
#endif

	float4 _hit = hits[width * y + x];
	if (_hit.w < 0)
	{
		return;
	}
	
	float3 hit = make_float3(_hit.x, _hit.y, _hit.z);
	
	int triangleIndex = roundf(_hit.w);
	float4 e1 = vertex(triangleIndex * 3 + 1);
	float4 e2 = vertex(triangleIndex * 3 + 2);
	float3 n;
	CROSS(n, e1, e2);
	if (DOT(n, camPos - hit) < 0) // normal should look at us
	{
		n = -n;
	}
	
	float3 size = max - min;
	float3 lm_voxel = make_float3(size.x / lm_div.x, size.y / lm_div.y, size.z / lm_div.z);
	
	int3 hit_cell = cell(hit, min, size, div);  // grid cell
	int3 lmp = nearest(_hit, min, size, lm_div); // light mesh point
	
	// WTF HERE?! __shared__ GOT SOME SHIT EXCEPT NEEDED RESULT!!!
	bits256 used;
	used.a[0]=used.a[1]=used.a[2]=used.a[3]=
	used.a[4]=used.a[5]=used.a[6]=used.a[7]=
	used.a[8]=used.a[9]=used.a[10]=used.a[11]=
	used.a[12]=used.a[13]=used.a[14]=used.a[15]=0;
	
	used.a[16]=used.a[17]=used.a[18]=used.a[19]=
	used.a[20]=used.a[21]=used.a[22]=used.a[23]=
	used.a[24]=used.a[25]=used.a[26]=used.a[27]=
	used.a[28]=used.a[29]=used.a[30]=used.a[31]=0;
	
	for (int j = 0; j < tpl_size; ++j)
	{
		int3 lmp2 = lmp + tpl[j];
		if (!in_bounds(lmp2, lm_div))
		{
			continue;
		}

		float3 pos = make_float3(min.x + lm_voxel.x * lmp2.x, min.y + lm_voxel.y * lmp2.y, min.z + lm_voxel.z * lmp2.z);
		
		if (len(pos - hit) > radius)
		{
			continue;
		}
		
		if (DOT(n, pos - hit) < 0) // light point is in the object, do not calculate it
		{
			continue;
		}
		
		float3 _hit = make_float3(hit.x, hit.y, hit.z);
		if (shortTest(min, max, div, _hit, hit_cell, pos, trisNum))
		{
			continue;
		}
		
		set_bit(used, j);
	}
	
	tpl_used[width * y + x] = used;
}

__host__ bits256 *shortTests
(
	const float4 *hits,
	const Scene& s,
	const int3& lm_div,
	const int3 *tpl, int tpl_size
)
{
	#if CPU
	
		bits256 *tpl_used = new bits256[s.width * s.height];
		memset(tpl_used, 0, sizeof(bits256) * s.width * s.height);
		for (int y = 0; y < s.height; ++y)
		{
			for (int x = 0; x < s.width; ++x)
			{
				shortTest(hits, s.width, s.height, x, y, s.min, s.max, s.div, lm_div, s.camPos, tpl, tpl_size, s.r, s.trisNum, tpl_used);
			}
		}
		return tpl_used;
	
	#else // GPU
	
		bits256 *tpl_used;
		CUDA_SAFE_CALL(cudaMalloc(&tpl_used, sizeof(bits256) * s.width * s.height));
		CUDA_SAFE_CALL(cudaMemset(tpl_used, 0, sizeof(bits256) * s.width * s.height));
		
		int block_x = BLOCK_X;
		int block_y = BLOCK_Y;
		int blocks_x = s.width  / block_x + (s.width  % block_x != 0);
		int blocks_y = s.height / block_y + (s.height % block_y != 0);

		shortTest<<<dim3(blocks_x, blocks_y), dim3(block_x, block_y)>>>(hits, s.width, s.height, s.min, s.max, s.div, lm_div, s.camPos, tpl, tpl_size, s.r, s.trisNum, tpl_used);
		CUDA_SAFE_CALL(cudaGetLastError());
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		
		return tpl_used;
	
	#endif
}

__host__ __device__ inline int3 xyz(int i, const int3& div)
{
	int x  = i  % div.x;
	int i2 = i  / div.x;
	int y  = i2 % div.y;
	int z  = i2 / div.y;
	return make_int3(x, y, z);
}

/*
 * CAST ONE LIGHT MESH POINTS SHADOW RAY (SOFT SHADOWS)
 */
__global__ void castLMShadow
(
	const int *points, int points_num,
	float3 min, float3 max, int3 div,
	int3 lm_div,
#if CPU
	int j,
#endif
	float3 lightPos,
	int trisNum,
	int *visible
)
{
#if GPU
	int j = (blockIdx.y * BLOCKS_ROW + blockIdx.x) * blockDim.x + threadIdx.x;
	if (j >= points_num)
	{
		return;
	}
#endif

	float3 size = max - min;
	float3 voxel = make_float3(size.x / lm_div.x, size.y / lm_div.y, size.z / lm_div.z);

	int3 p = xyz(points[j], lm_div);
	float3 pos = make_float3(min.x + voxel.x * p.x, min.y + voxel.y * p.y, min.z + voxel.z * p.z);
	
	float3 dir = lightPos - pos;
	float dist = len(dir);
	dir /= dist;
	
	visible[j] = !rayScene<TEST>(min, max, div, pos, dir, dist, trisNum, NULL);
}

/*
 * CAST LIGHT MESH POINT SHADOW RAYS (SOFT SHADOWS)
 */
__host__ void castLMShadowRays
(
	const int *points, int points_num,
	const Scene& s,
	const int3& lm_div,
	const float3& lightPos,
	int *visible
)
{
	float3 size = s.max - s.min;
	float3 voxel = make_float3(size.x / lm_div.x, size.y / lm_div.y, size.z / lm_div.z);
	
	#if CPU

		for (int j = 0; j < points_num; ++j)
		{
			castLMShadow(points, points_num, s.min, s.max, s.div, lm_div, j, lightPos, s.trisNum, visible);
		}
	
	#else // GPU

		int block = BLOCK;      // threads per block
		int blocks = points_num / block + (points_num % block != 0);
		int blocks_row_max = BLOCKS_ROW;
		int blocks_rows = blocks / blocks_row_max + (blocks % blocks_row_max != 0);
		int blocks_row = (blocks_rows > 1) ? blocks_row_max : blocks;
		
		castLMShadow<<<dim3(blocks_row, blocks_rows), block>>>(points, points_num, s.min, s.max, s.div, lm_div, lightPos, s.trisNum, visible);
		CUDA_SAFE_CALL(cudaGetLastError());

	#endif
}

__global__ void calcPixel
(
	const float4 *hits,
	const int3 *tpl, int tpl_size, float radius,
	const bits256 *tpl_used,
	const int *lm,
	int width, int height,
#if CPU
	int x, int y,
#endif
	float3 min, float3 max,
	float3 camPos,
	int3 lm_div,
	int light, float3 lightPos,
	float3 background,
	int trisNum,
	float3 *img
)
{
#if GPU
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
#endif
	
	float4 hit = hits[y * width + x];
	if (hit.w < 0) // primary ray missed the scene
	{
		img[y * width + x] = background;
		return;
	}
	
	int triangleIndex = roundf(hit.w);
	float4 e1 = vertex(triangleIndex * 3 + 1);
	float4 e2 = vertex(triangleIndex * 3 + 2);
	float3 n;
	CROSS(n, e1, e2);
	if (DOT(n, camPos - hit) < 0) // normal should look at us
	{
		n = -n;
	}
	normalize(n);
	
	float3 size = max - min;
	int3 lmp = nearest(hit, min, size, lm_div);
	
	float3 toLight = lightPos - hit;
	normalize(toLight);
	if (DOT(n, toLight) < 0) // on the dark side
	{
		return;
	}
	
	float3 lm_voxel = make_float3(size.x / lm_div.x, size.y / lm_div.y, size.z / lm_div.z);
	
	float3 add = make_float3(0,0,0);
	
	float sum = 0;
	float total = 0;
	for (int j = 0; j < tpl_size; ++j)
	{
		int3 lmp2 = lmp + tpl[j];
		if (!get_bit(tpl_used[y * width + x], j))
		{
			continue;
		}
		float3 pos = make_float3(min.x + lm_voxel.x * lmp2.x, min.y + lm_voxel.y * lmp2.y, min.z + lm_voxel.z * lmp2.z);
		float length = len(pos - hit);
		float add = expf(-(length * length) / (radius * radius));
		total += add;
		if (get_bit(lm[index(lmp2, lm_div)], light))
		{
			sum += add;
		}
	}
	if (total > 0)
	{
		add += make_float3(1, 1, 1) * DOT(n, toLight) * (sum / total);
	}
	
	img[y * width + x] += add;
}

__host__ float3 *calcPixels
(
	const float4 *hits,
	const int3 *tpl, int tpl_size, const bits256 *tpl_used,
	const int *lm,
	const Scene& s,
	const int3& lm_div
)
{
	#if CPU
	
		float3 *img = new float3[s.width * s.height];
		memset(img, 0, sizeof(float3) * s.width * s.height);
	
		for (int y = 0; y < s.height; ++y)
		{
			for (int x = 0; x < s.width; ++x)
			{
				for (int i = 0; i < s.lightsNum; ++i)
				{
					calcPixel(hits, tpl, tpl_size, s.r, tpl_used, lm, s.width, s.height, x, y, s.min, s.max, s.camPos, lm_div, i, s.lights[i].pos, s.background, s.trisNum, img);
				}
			}
		}
		
		return img;
	
	#else // GPU
	
		float3 *_img;
		CUDA_SAFE_CALL(cudaMalloc(&_img, sizeof(float3) * s.width * s.height));
		CUDA_SAFE_CALL(cudaMemset(_img, 0, sizeof(float3) * s.width * s.height));
		
		int block_x = BLOCK_X;
		int block_y = BLOCK_Y;
		int blocks_x = s.width  / block_x + (s.width  % block_x != 0);
		int blocks_y = s.height / block_y + (s.height % block_y != 0);
		
		for (int i = 0; i < s.lightsNum; ++i)
		{
			calcPixel<<<dim3(blocks_x, blocks_y), dim3(block_x, block_y)>>>(hits, tpl, tpl_size, s.r, tpl_used, lm, s.width, s.height, s.min, s.max, s.camPos, lm_div, i, s.lights[i].pos, s.background, s.trisNum, _img);
			CUDA_SAFE_CALL(cudaGetLastError());
		}
		
		float3 *img = new float3[s.width * s.height];
		CUDA_SAFE_CALL(cudaMemcpy(img, _img, sizeof(float3) * s.width * s.height, cudaMemcpyDeviceToHost));
		
		return img;
	
	#endif
}

#endif // CPU_H
