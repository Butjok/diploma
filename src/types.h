#ifndef TYPES_H
#define TYPES_H

#include "common.h"

#if CPU
inline float3 make_float3(float x, float y, float z)
{
	float3 r;
	r.x = x;
	r.y = y;
	r.z = z;
	return r;
}
inline float4 make_float4(float x, float y, float z, float w)
{
	float4 r;
	r.x = x;
	r.y = y;
	r.z = z;
	r.w = w;
	return r;
}
inline int3 make_int3(int x, int y, int z)
{
	int3 r;
	r.x = x;
	r.y = y;
	r.z = z;
	return r;
}
inline int4 make_int4(int x, int y, int z, int w)
{
	int4 r;
	r.x = x;
	r.y = y;
	r.z = z;
	r.w = w;
	return r;
}
#endif

__host__ __device__ inline float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

// float3 float3
__host__ __device__ inline float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// float3 float4 
__host__ __device__ inline float3 operator+(const float3& a, const float4& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float4& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// float4 float3
__host__ __device__ inline float3 operator+(const float4& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float4& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// float4 float4 
__host__ __device__ inline float4 operator+(const float4& a, const float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline float4 operator-(const float4& a, const float4& b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
__host__ __device__ inline float3& operator-=(float3& a, const float3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
__host__ __device__ inline float3 operator*(const float3& a, const float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline float3 operator/(const float3& a, const float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
__host__ __device__ inline float3& operator*=(float3& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}
__host__ __device__ inline float3& operator/=(float3& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	return a;
}
__host__ __device__ inline float at(const float3& a, const int i)
{
	return i == 0 ? a.x : i == 1 ? a.y : a.z;
}
__host__ __device__ inline float& at(float3& a, const int i)
{
	return i == 0 ? a.x : i == 1 ? a.y : a.z;
}
__host__ __device__ inline float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 cross(const float3& a, const float3& b)
{
	return make_float3
	(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}
__host__ __device__ inline float len(const float3& a)
{
	return sqrtf(dot(a, a));
}
__host__ __device__ inline void normalize(float3& a)
{
	float l = len(a);
	a.x /= l;
	a.y /= l;
	a.z /= l;
}
__host__ __device__ inline float3 operator*(const float3& a, const float3& b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline int3 operator+(const int3& a, const int3& b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

struct Light
{
	float3 pos, color;
};

struct Tri
{
	float4 v0, e1, e2;
};

struct Scene
{
	float3 camPos, camDir, camUp;
	float hFov, vFov;
	int width, height;
	int traceDepth;
	int3 div;
	int maxLevels;
	int soft;
	float h, r;
	float gamma;
	float3 background;
	float3 ambient;
	int lightsNum;
	Light *lights;
	int trisNum;
	Tri *tris;
	float3 min, max;
};

struct bits256
{
	// char a[32]; // 32 bytes * 8 bits = 256 bits
	int a[32];
};

#endif // TYPES_H
