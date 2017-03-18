#ifndef TESTS_H
#define TESTS_H

#include "common.h"

#include "types.h"

#define DOT(a, b) ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)
#define CROSS(c, a, b)				\
{									\
	c.x = a.y * b.z - a.z * b.y;	\
	c.y = a.z * b.x - a.x * b.z;	\
	c.z = a.x * b.y - a.y * b.x;	\
}
#define MINUS(c, a, b)	\
{						\
	c.x = a.x - b.x;	\
	c.y = a.y - b.y;	\
	c.z = a.z - b.z;	\
}

__host__ __device__ inline float3 minus(const float3& a, const float4& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 minus(const float4& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline int rayTriTest(const float3& orig, const float3& dir, const float4& v0, const float4& e1, const float4& e2, float *t)
{
//	const float3 pvec = cross(dir, e2);
	float3 pvec;
	CROSS(pvec, dir, e2);
	const float invdet = 1.0f / DOT(e1, pvec);
//	const float3 tvec = orig - v0;
	float3 tvec;
	MINUS(tvec, orig, v0);
//	const float3 qvec = cross(tvec, e1);
	float3 qvec;
	CROSS(qvec, tvec, e1);
	const float u = DOT(tvec, pvec) * invdet;
	const float v = DOT(dir, qvec) * invdet;
	*t = DOT(e2, qvec) * invdet;
	return !(u < 0 || u > 1 || v < 0 || u + v > 1);
}

__device__ inline int rayBoxTest(
	const float3& orig, const float3& dir,
	const float3& min,  const float3& max,
	float *_tmin)
{
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	
	if (dir.x >= 0)
	{
		tmin = (min.x - orig.x) / dir.x;
		tmax = (max.x - orig.x) / dir.x;
	}
	else
	{
		tmin = (max.x - orig.x) / dir.x;
		tmax = (min.x - orig.x) / dir.x;
	}

	if (dir.y >= 0)
	{
		tymin = (min.y - orig.y) / dir.y;
		tymax = (max.y - orig.y) / dir.y;
	}
	else
	{
		tymin = (max.y - orig.y) / dir.y;
		tymax = (min.y - orig.y) / dir.y;
	}

	if (tmin > tymax || tymin > tmax)
	{
		return 0;
	}
	if (tymin > tmin)
	{
		tmin = tymin;
	}
	if (tymax < tmax)
	{
		tmax = tymax;
	}

	if (dir.z >= 0)
	{
		tzmin = (min.z - orig.z) / dir.z;
		tzmax = (max.z - orig.z) / dir.z;
	}
	else
	{
		tzmin = (max.z - orig.z) / dir.z;
		tzmax = (min.z - orig.z) / dir.z;
	}

	if (tmin > tzmax || tzmin > tmax)
	{
		return 0;
	}
	if (tzmin > tmin)
	{
		tmin = tzmin;
	}
	if (tzmax < tmax)
	{
		tmax = tzmax;
	}

	*_tmin = tmin;

	return tmax > 0;
}

// Copyright Yelagin :)
__host__ inline int triBoxTest(const Tri& t, const float3& min, const float3& max)
{
	float3 bCenter = (min + max) / 2;
	float3 bSize = (max - min) / 2;

	float3 tv0 = make_float3(t.v0.x, t.v0.y, t.v0.z);
	float3 te1 = make_float3(t.e1.x, t.e1.y, t.e1.z);
	float3 te2 = make_float3(t.e2.x, t.e2.y, t.e2.z);

	float3 v[3] = {tv0 - bCenter, te1 + tv0 - bCenter, te2 + tv0 - bCenter};
	float3 eAbs[3] = {
		make_float3(fabs(v[1].x - v[0].x), fabs(v[1].y - v[0].y), fabs(v[1].z - v[0].z)),
		make_float3(fabs(v[2].x - v[1].x), fabs(v[2].y - v[1].y), fabs(v[2].z - v[1].z)),
		make_float3(fabs(v[0].x - v[2].x), fabs(v[0].y - v[2].y), fabs(v[0].z - v[2].z))
	};
	float3 crosses[3] = {cross(v[0], v[1]), cross(v[1], v[2]), cross(v[2], v[0])};
	float3 across[3] = {-crosses[1]-crosses[2], -crosses[2]-crosses[0], -crosses[0]-crosses[1]};
	for (int j=0; j<3; j++) {
		double minProj;
		double maxProj;
		int j1 = (j+1)%3;
		int j2 = (j+2)%3;
		for (int i=0; i<3; i++) {
			if (at(crosses[i], j) < at(across[i], j)) {
				minProj = at(crosses[i], j);
				maxProj = at(across[i], j);
			} else {
				minProj = at(across[i], j);
				maxProj = at(crosses[i], j);
			}
			double bProj = at(eAbs[i],j1) * at(bSize,j2) + at(eAbs[i],j2) * at(bSize,j1);
			if (minProj > bProj || maxProj < -bProj) return false;
		}
		if (at(v[1],j) < at(v[0],j)) {
			minProj = at(v[1],j);
			maxProj = at(v[0],j);
		} else {
			minProj = at(v[0],j);
			maxProj = at(v[1],j);
		}
		if (at(v[2],j) < minProj) {
			minProj = at(v[2],j);
		} else if (at(v[2],j) > maxProj) {
			maxProj = at(v[2],j);
		}
		if (minProj > at(bSize,j) || maxProj < -at(bSize,j)) return false;
	}
	float3 normal = crosses[0] - across[0];
	float3 corner;
	for (int i=0; i<3; i++) at(corner,i) = (at(normal,i) > 0) ? at(bSize,i) : -at(bSize,i);
	return dot(normal, (v[0] + corner)) >= 0 && dot(normal, (v[0] - corner)) <= 0;
}

#endif // TESTS_H
