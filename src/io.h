#ifndef IO_H
#define IO_H

#include <stdio.h>

#include "types.h"

#include "io.h"

#include "tests.h"
#include "utils.h"

inline static void readi3(FILE *in, int3 *v)
{
	fscanf(in, "%d %d %d", &v->x, &v->y, &v->z);
}
inline static void readf3(FILE *in, float3 *v)
{
	fscanf(in, "%f %f %f", &v->x, &v->y, &v->z);
}
inline static void readf3(FILE *in, float4 *v)
{
	fscanf(in, "%f %f %f", &v->x, &v->y, &v->z);
}
void read(FILE *in, Scene *s)
{
	float3 at, up;

	readf3(in, &s->camPos);
	readf3(in, &at);
	readf3(in, &up);

	s->camDir = at - s->camPos;
	normalize(s->camDir);

	normalize(up);
	float3 right = cross(s->camDir, up);
	normalize(right);
	s->camUp = cross(right, s->camDir);
	normalize(s->camUp);

	fscanf(in, "%f %f", &s->hFov, &s->vFov);

	s->hFov = s->hFov * M_PI / 180.0f; // degrees to radians
	s->vFov = s->vFov * M_PI / 180.0f;

	fscanf(in, "%d %d", &s->width, &s->height);
	fscanf(in, "%d", &s->traceDepth);
	readi3(in, &s->div);
	fscanf(in, "%d", &s->maxLevels);
	fscanf(in, "%d %f %f", &s->soft, &s->h, &s->r);
	fscanf(in, "%f", &s->gamma);
	readf3(in, &s->background);
	readf3(in, &s->ambient);

	fscanf(in, "%d", &s->lightsNum);
	s->lights = new Light[s->lightsNum];
	for (int i = 0; i < s->lightsNum; ++i)
	{
		readf3(in, &s->lights[i].pos);
		readf3(in, &s->lights[i].color);
	}

	int first = 1;

	fscanf(in, "%d", &s->trisNum);
	s->tris = new Tri[s->trisNum];
	for (int i = 0; i < s->trisNum; ++i)
	{
		Tri *tri = s->tris + i;
		float3 v0, v1, v2;

		readf3(in, &v0);
		readf3(in, &v1);
		readf3(in, &v2);

		float3 e1 = v1 - v0;
		float3 e2 = v2 - v0;

		tri->v0 = make_float4(v0.x, v0.y, v0.z, 0);
		tri->e1 = make_float4(e1.x, e1.y, e1.z, 0);
		tri->e2 = make_float4(e2.x, e2.y, e2.z, 0);

		float3 trash3;
		readf3(in, &trash3);
		readf3(in, &trash3);
		readf3(in, &trash3);

		float trash;
		fscanf(in, "%f", &trash);

		if (first)
		{
			first = 0;

			s->min.x = minf3(v0.x, v1.x, v2.x);
			s->min.y = minf3(v0.y, v1.y, v2.y);
			s->min.z = minf3(v0.z, v1.z, v2.z);

			s->max.x = maxf3(v0.x, v1.x, v2.x);
			s->max.y = maxf3(v0.y, v1.y, v2.y);
			s->max.z = maxf3(v0.z, v1.z, v2.z);
		}
		else
		{
			s->min.x = minf4(s->min.x, v0.x, v1.x, v2.x);
			s->min.y = minf4(s->min.y, v0.y, v1.y, v2.y);
			s->min.z = minf4(s->min.z, v0.z, v1.z, v2.z);

			s->max.x = maxf4(s->max.x, v0.x, v1.x, v2.x);
			s->max.y = maxf4(s->max.y, v0.y, v1.y, v2.y);
			s->max.z = maxf4(s->max.z, v0.z, v1.z, v2.z);
		}
	}
	
	float3 size = s->max - s->min;
	float max = maxf3(size.x, size.y, size.z);

	s->min -= make_float3(0.5f, 0.5f, 0.5f);
	s->max += make_float3(0.5f, 0.5f, 0.5f);
	if (s->soft)
	{
		s->min -= make_float3(s->r, s->r, s->r);
		s->max += make_float3(s->r, s->r, s->r);
	}
}

inline static int sat(float a)
{
	return a < 0 ? 0 : a > 255 ? 255 : a;
}
void write(FILE *out, int width, int height, const float3 *img)
{
	int pad = (width * 3) % 4 == 0 ? 0 : 4 - (width * 3) % 4;
	int pitch = width * 3 + pad;
	int size  = 54 + pitch * height;
	int ssize = pitch * height;

	// BMP header
	putc('B', out);
	putc('M', out);
	fwrite(&size, 4, 1, out);
	putc(0, out); putc(0, out);
	putc(0, out); putc(0, out);
	putc(54, out); putc(0, out); putc(0, out); putc(0, out);

	// DIB header
	putc(40, out); putc(0, out); putc(0, out); putc(0, out);
	fwrite(&width, 4, 1, out);
	fwrite(&height, 4, 1, out);
	putc(1, out); putc(0, out);
	putc(24, out); putc(0, out);
	putc(0, out); putc(0, out); putc(0, out); putc(0, out);
	fwrite(&ssize, 4, 1, out);
	putc(0, out); putc(0, out); putc(0, out); putc(0, out);
	putc(0, out); putc(0, out); putc(0, out); putc(0, out);
	putc(0, out); putc(0, out); putc(0, out); putc(0, out);
	putc(0, out); putc(0, out); putc(0, out); putc(0, out);

	for (int y = height - 1; y >= 0; --y)
	{
		for (int x = 0; x < width; ++x)
		{
			const float3& color = img[width * y + x];
			putc(sat(color.z), out);
			putc(sat(color.y), out);
			putc(sat(color.x), out);
		}
		for (int i = 0; i < pad; ++i)
		{
			putc(0, out);
		}
	}
}

#endif // IO_H
