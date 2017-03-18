#include <stdio.h>
#include <math.h>
#include <time.h>

#include "common.h"
#include "utils.h"

#if GPU
#include <cuda_runtime_api.h>
#include <cutil.h>
#endif

#include "io.h"
#include "grid.h"
#include "flatgrid.h"
#include "kernels.h"

inline void printf3(float3 v)
{
	printf("(%.3f, %.3f, %.3f)\n", v.x, v.y, v.z);
}
void print_scene(const Scene s)
{
	printf("Position:      "); printf3(s.camPos);
	printf("Direction:     "); printf3(s.camDir);
	// printf("Up:            "); printf3(s.camUp);
	printf("FOV (degrees): %.3f x %.3f\n", s.hFov / M_PI * 180.0f, s.vFov / M_PI * 180.0f);
	printf("Resolution:    %d x %d\n", s.width, s.height);
	// printf("Trace depth:   %d\n", s.traceDepth);
	#if USE_GRID
	{
		printf("Grid:          YES, %d x %d x %d (max %d levels)\n", s.div.x, s.div.y, s.div.z, s.maxLevels);
	}
	#else
	{
		printf("Grid:          NO\n");
	}
	#endif
	if (s.soft)
	{
		printf("Soft shadows:  YES, h = %.3f, r = %.3f, r/h = %.3f\n", s.h, s.r, s.r / s.h);
	}
	else
	{
		printf("Soft shadows:  NO\n");
	}
	printf("Gamma:         %.3f\n", s.gamma);
	printf("Background:    "); printf3(s.background);
	printf("Ambient:       "); printf3(s.ambient);
	printf("Lights:        %d\n", s.lightsNum);
	printf("Triangles:     %d\n", s.trisNum);
	printf("Scene min:     "); printf3(s.min);
	printf("Scene max:     "); printf3(s.max);
	printf("Scene size:    "); printf3(s.max - s.min);
	printf("\n");
}

inline float3 normal(int triangleIndex, const float3& camPos)
{
	float4 v0 = ___tris[triangleIndex * 3];
	float4 e1 = ___tris[triangleIndex * 3 + 1];
	float4 e2 = ___tris[triangleIndex * 3 + 2];
	float3 n;
	CROSS(n, e1, e2);
	if (DOT(n, camPos - v0) < 0) // normal should look at us
	{
		n = -n;
	}
	normalize(n);
	return n;
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("Usage: %s in.txt\n", argv[0]);
		return 0;
	}
	
	Scene s;

	/*
	 * Load scene
	 */
	{
		FILE *in = fopen(argv[1], "rt");
		if (!in)
		{
			perror(argv[1]);
			return 1;
		}
		read(in, &s);
		fclose(in);
	}

	/*
	 * Fill grid
	 */
	int tris_num;
	int grid_size;
	
	#if !USE_GRID
	{
		tris_num = s.trisNum;
		___tris = (float4 *) s.tris;
	}
	#else
	{
		Grid __grid(s.div.x, s.div.y, s.div.z);
		for (int i = 0; i < s.trisNum; ++i)
		{
			__grid.add(s.tris[i], i, s.min, s.max);
		}
		
		float3 size = s.max - s.min;
		float k = powf(9.0f / (size.x * size.y * size.z), 1.0f / 3.0f);
		float3 alpha = size * k;
		__grid.extend(s.tris, 0, s.maxLevels, alpha, s.min, s.max);

		std::vector<int> _grid;
		std::vector<float4> _tris;
		makeFlatGrid(&__grid, &_grid, (float4 *) s.tris, &_tris, 0);
		
		grid_size = _grid.size();
		___grid = new int[grid_size];
		tris_num = _tris.size() / 3;
		___tris = new float4[tris_num * 3];

		int j;
		
		std::vector<int>::const_iterator i;
		for (i = _grid.begin(), j = 0; i != _grid.end(); ++i, ++j)
		{
			___grid[j] = *i;
		}
		
		std::vector<float4>::const_iterator i2;
		for (i2 = _tris.begin(), j = 0; i2 != _tris.end(); ++i2, ++j)
		{
			___tris[j] = *i2;
		}
	}
	#endif

	/*
	 * Info
	 */
	print_scene(s);
	
	#if GPU
	{
		// device triangles
		float4 *_tris;
		CUDA_SAFE_CALL(cudaMalloc(&_tris, sizeof(Tri) * tris_num));
		CUDA_SAFE_CALL(cudaMemcpy(_tris, ___tris, sizeof(Tri) * tris_num, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaBindTexture(NULL, texTris, _tris));
		
		#if USE_GRID
		{
			// device grid
			int *_grid;
			CUDA_SAFE_CALL(cudaMalloc(&_grid, sizeof(int) * grid_size));
			CUDA_SAFE_CALL(cudaMemcpy(_grid, ___grid, sizeof(int) * grid_size, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaBindTexture(NULL, texGrid, _grid));
		}
		#endif
	}	
	#endif
	
	/*
	 * !!! ATTENTION !!! Should use CudaTimer!
	 */
	Timer total, subtimer;
	// CudaTimer timer;
	Timer timer;
	total.start();

	/*
	 * CAST PRIMARY RAYS
	 */
	float4 *_hits = castPrimaryRays(s);	// device
	float4 *hits;						// host
	
	// device -> host
	#if CPU
		hits = _hits;
	#else // GPU
		hits = new float4[s.width * s.height];
		CUDA_SAFE_CALL(cudaMemcpy(hits, _hits, sizeof(float4) * s.width * s.height, cudaMemcpyDeviceToHost));
	#endif
	
	float3 *img;
	
	/*
	 * SHARP SHADOWS
	 */
	if (!s.soft)
	{
		/*
		 * CAST SHADOW RAYS
		 */
		int *_visible;	// device
		int *visible;	// host
		#if CPU
			_visible = new int[s.lightsNum * s.width * s.height];
		#else // GPU
			CUDA_SAFE_CALL(cudaMalloc(&_visible, sizeof(int) * s.lightsNum * s.width * s.height));
		#endif
		
		// this loop's iterations should be parallel
		timer.start();
		for (int i = 0; i < s.lightsNum; ++i)
		{
			castShadowRays(_hits, s, s.lights[i].pos, _visible + i * s.width * s.height);
		}
		#if GPU
			CUDA_SAFE_CALL(cudaThreadSynchronize());
		#endif
		timer.stop();
		printf("Shadow rays: %.2f sec\n\n", timer.time());
		
		// device -> host
		#if CPU
			visible = _visible;
		#else // GPU
			visible = new int[s.lightsNum * s.width * s.height];
			CUDA_SAFE_CALL(cudaMemcpy(visible, _visible, sizeof(int) * s.lightsNum * s.width * s.height, cudaMemcpyDeviceToHost));
		#endif
	
		/*
		 * CALCULATE PIXEL COLORS
		 */
		img = new float3[s.width * s.height];
		for (int y = 0; y < s.height; ++y)
		{
			for (int x = 0; x < s.width; ++x)
			{
				float4 hit = hits[y * s.width + x];
				int triangleIndex = roundf(hit.w);
				if (triangleIndex == -1) // primary ray missed the scene
				{
					img[y * s.width + x] = s.background;
					continue;
				}
				float3 n = normal(triangleIndex, s.camPos);
				for (int i = 0; i < s.lightsNum; ++i)
				{
					if (!visible[i * s.width * s.height + y * s.width + x])
					{
						continue;
					}
					// TODO: calculate color by triangle's material
					float3 toLight = s.lights[i].pos - hit;
					normalize(toLight);
					img[y * s.width + x] += make_float3(1, 1, 1) * DOT(n, toLight);
				}
			}
		}

	/*
	 * SOFT SHADOWS
	 */
	}
	else
	{
		float3 size = s.max - s.min;
		int3 lm_div;
		lm_div.x = (int) (size.x / s.h + 0.5f);
		lm_div.y = (int) (size.y / s.h + 0.5f);
		lm_div.z = (int) (size.z / s.h + 0.5f);
		float3 lm_voxel = make_float3(size.x / lm_div.x, size.y / lm_div.y, size.z / lm_div.z);

		int3 *tpl;
		int tpl_size;
	
		/*
		 * TEMPLATE INITIALIZATION
		 */
		{
			int half = (int) (s.r / s.h) + 1; // с запасом
			int one = half + 1 + half;
		
			tpl = new int3[one * one * one]; // 111oneoneone :)
			tpl_size = 0;
			
			float h = s.h;
			float3 poses[8] =
			{
				make_float3(0, 0, 0),
				make_float3(h, 0, 0),
				make_float3(0, h, 0),
				make_float3(0, 0, h),
				make_float3(h, h, 0),
				make_float3(0, h, h),
				make_float3(h, 0, h),
				make_float3(h, h, h)
			};
			
			for (int z = -half; z <= half; ++z)
			for (int y = -half; y <= half; ++y)
			for (int x = -half; x <= half; ++x)
			{
				float3 lm_pos = make_float3(lm_voxel.x * x, lm_voxel.y * y, lm_voxel.z * z);
				for (int i = 0; i < 8; ++i)
				{
					if (len(lm_pos - poses[i]) < s.r + 0.001f)
					{
						tpl[tpl_size++] = make_int3(x, y, z);
						break;
					}
				}
			}
		}
		{
			printf("Light grids:    %d x %d x %d\n", lm_div.x, lm_div.y, lm_div.z);
			printf("In sphere:      %d\n\n", tpl_size);
		}
		
		/*
		 * DETERMINE USED LIGHT MESH POINTS
		 */
		int3 *_tpl;
		#if CPU
			_tpl = tpl;
		#else // GPU
			CUDA_SAFE_CALL(cudaMalloc(&_tpl, sizeof(int3) * tpl_size));
			CUDA_SAFE_CALL(cudaMemcpy(_tpl, tpl, sizeof(int3) * tpl_size, cudaMemcpyHostToDevice));
		#endif
		
		// used per pixel light mesh points
		timer.start();
		bits256 *_tpl_used = shortTests(_hits, s, lm_div, _tpl, tpl_size);	// device
		bits256 *tpl_used;													// host
		timer.stop();
		{
			printf("Short tests:    %.2f sec\n", timer.time());
		}
		
		// device -> host
		#if CPU
			tpl_used = _tpl_used;
		#else // GPU
			tpl_used = new bits256[s.width * s.height];
			CUDA_SAFE_CALL(cudaMemcpy(tpl_used, _tpl_used, sizeof(bits256) * s.width * s.height, cudaMemcpyDeviceToHost));
		#endif
		
		int *lm_points = new int[lm_div.x * lm_div.y * lm_div.z];
		int lm_points_num = 0;
		
		// used (global) light mesh pixels
		{
			char *lm = new char[lm_div.x * lm_div.y * lm_div.z];
			memset(lm, 0, lm_div.x * lm_div.y * lm_div.z);
		
			subtimer.start();
			for (int y = 0; y < s.height; ++y)
			{
				for (int x = 0; x < s.width; ++x)
				{
					if (hits[s.width * y + x].w < 0)
					{
						continue;
					}
					int3 lmp = nearest(hits[s.width * y + x], s.min, size, lm_div);
					for (int j = 0; j < tpl_size; ++j)
					{
						int3 lmp2 = lmp + tpl[j];
						if (!in_bounds(lmp2, lm_div))
						{
							continue;
						}
						if (!get_bit(tpl_used[s.width * y + x], j))
						{
							continue;
						}
						int ind = index(lmp2, lm_div);
						if (!lm[ind])
						{
							lm[ind] = 1;
							lm_points[lm_points_num++] = ind;
						}
					}
				}
			}
			subtimer.stop();
		}
		{
			printf("LM points calc: %.2f\n", subtimer.time());
			printf("LM points:      %d\n", lm_points_num);
		}
		

		/*
		 * CAST LIGHT MESH POINTS SHADOW RAYS
		 */
		int *_lm_points;
		int *_visible;
		#if CPU
			_lm_points = lm_points;
			_visible = new int[s.lightsNum * lm_points_num];
		#else
			CUDA_SAFE_CALL(cudaMalloc(&_lm_points, sizeof(int) * lm_points_num));
			CUDA_SAFE_CALL(cudaMemcpy(_lm_points, lm_points, sizeof(int) * lm_points_num, cudaMemcpyHostToDevice));
			
			CUDA_SAFE_CALL(cudaMalloc(&_visible, sizeof(int) * s.lightsNum * lm_points_num));
			CUDA_SAFE_CALL(cudaMemset(_visible, 0, sizeof(int) * s.lightsNum * lm_points_num));
		#endif
		
		timer.start();
		for (int i = 0; i < s.lightsNum; ++i)
		{
			castLMShadowRays(_lm_points, lm_points_num, s, lm_div, s.lights[i].pos, _visible + i * lm_points_num);
		}
		#if GPU
			CUDA_SAFE_CALL(cudaThreadSynchronize());
		#endif
		timer.stop();
		{
			printf("LM shadow rays: %.2f sec\n\n", timer.time());
		}

		int *visible;
		#if CPU
			visible = _visible;
		#else // GPU
			visible = new int[s.lightsNum * lm_points_num];
			CUDA_SAFE_CALL(cudaMemcpy(visible, _visible, sizeof(int) * s.lightsNum * lm_points_num, cudaMemcpyDeviceToHost));
		#endif

		/*
		 * COMPLETE LIGHT MESH
		 */
		int *lm = new int[lm_div.x * lm_div.y * lm_div.z];
		memset(lm, 0, sizeof(int) * lm_div.x * lm_div.y * lm_div.z);
		for (int j = 0; j < lm_points_num; ++j)
		{
			int used = 0;
			for (int i = 0; i < s.lightsNum; ++i)
			{
				if (visible[i * lm_points_num + j])
				{
					set_bit(used, i);
				}
			}
			lm[lm_points[j]] = used;
		}
		int *_lm;
		#if CPU
			_lm = lm;
		#else // GPU
			CUDA_SAFE_CALL(cudaMalloc(&_lm, sizeof(int) * lm_div.x * lm_div.y * lm_div.z));
			CUDA_SAFE_CALL(cudaMemcpy(_lm, lm, sizeof(int) * lm_div.x * lm_div.y * lm_div.z, cudaMemcpyHostToDevice));
		#endif
		
		/*
		 * CALCULATE PIXEL COLORS
		 */
		timer.start();
		img = calcPixels(_hits, _tpl, tpl_size, _tpl_used, _lm, s, lm_div);
		timer.stop();
		{
			printf("Pixel calc: %.2f sec\n\n", timer.time());
		}

	} // SHADOW TYPE SELECTION
	
	total.stop();
	float total_time = total.time();
	{
		printf("---\n");
		printf("Total: %.2f sec\n\n", total_time);
	}

	/*
	 * Image correction (normalization, gamma)
	 */
	{
		float max = maxf3(s.background.x, s.background.y, s.background.z);
		for (int y = 0; y < s.height; ++y)
		{
			for (int x = 0; x < s.width; ++x)
			{
				const float3& color = img[s.width * y + x];
				max = maxf4(max, color.x, color.y, color.z);
			}
		}
		for (int y = 0; y < s.height; ++y)
		{
			for (int x = 0; x < s.width; ++x)
			{
				float3& color = img[s.width * y + x];

				color.x /= max;
				color.y /= max;
				color.z /= max;

				color.x = pow(color.x, s.gamma);
				color.y = pow(color.y, s.gamma);
				color.z = pow(color.z, s.gamma);

				color.x *= 255;
				color.y *= 255;
				color.z *= 255;
			}
		}
	}

	/*
	 * Write image
	 */
	{
		const char *device;
		#if CPU
			device = "CPU";
		#else
			device = "GPU";
		#endif
		char *shadows;
		if (!s.soft)
		{
			shadows = "sharp";
		}
		else
		{
			shadows = new char[1024];
			snprintf(shadows, 1024, "r:h-%.1f", s.r / s.h);
		}
		char filename[1024];
		snprintf(filename, sizeof(filename), "out/%d -- %s -- %s -- %s -- %d lights -- %.2f.bmp", time(NULL), argv[1], device, shadows, s.lightsNum, total_time);
	
		FILE *out = fopen(filename, "wb");
		if (!out)
		{
			perror(argv[2]);
			return 1;
		}
		write(out, s.width, s.height, img);
		fclose(out);
	}
	
	printf(":)\n");

	return 0;
}
