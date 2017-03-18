#ifndef GRID_H
#define GRID_H

#include <vector>

#include "types.h"
#include "tests.h"
#include "utils.h"

#include "tribox.h"

#define CELL 0
#define GRID 1

#define THRESHOLD 15

struct Node
{
	virtual int type() const = 0;
	virtual ~Node() {}
};

struct Cell : Node
{
	int type() const { return CELL; }
	std::vector<int> indexes;
};

struct Grid : Node
{
	int type() const { return GRID; }
	int divX, divY, divZ;
	Node **nodes;

	Grid(int divX, int divY, int divZ)
		: divX(divX), divY(divY), divZ(divZ)
	{
		nodes = (Node **) calloc(sizeof(Node *), divX * divY * divZ);
	}

	const Node *at(int x, int y, int z) const
	{
		return nodes[divX * divY * z + divX * y + x];
	}
	Node *& at(int x, int y, int z)
	{
		return nodes[divX * divY * z + divX * y + x];
	}

	void add(const Tri& tri, int index, const float3& min, const float3& max)
	{
		const float3 triv0 = make_float3(tri.v0.x, tri.v0.y, tri.v0.z);
		const float3 trie1 = make_float3(tri.e1.x, tri.e1.y, tri.e1.z);
		const float3 trie2 = make_float3(tri.e2.x, tri.e2.y, tri.e2.z);

		const float3 triMin = make_float3
		(
			minf3(triv0.x, (triv0 + trie1).x, (triv0 + trie2).x),
			minf3(triv0.y, (triv0 + trie1).y, (triv0 + trie2).y),
			minf3(triv0.z, (triv0 + trie1).z, (triv0 + trie2).z)
		);
		const float3 triMax = make_float3
		(
			maxf3(triv0.x, (triv0 + trie1).x, (triv0 + trie2).x),
			maxf3(triv0.y, (triv0 + trie1).y, (triv0 + trie2).y),
			maxf3(triv0.z, (triv0 + trie1).z, (triv0 + trie2).z)
		);

		float3 size = max - min;
		float3 voxel = make_float3(size.x / divX, size.y / divY, size.z / divZ);

		int xMin = maxi2(int((triMin.x - min.x) / voxel.x) - 1, 0);
		int yMin = maxi2(int((triMin.y - min.y) / voxel.y) - 1, 0);
		int zMin = maxi2(int((triMin.z - min.z) / voxel.z) - 1, 0);

		int xMax = mini2(int((triMax.x - min.x) / voxel.x) + 1, divX - 1);
		int yMax = mini2(int((triMax.y - min.y) / voxel.y) + 1, divY - 1);
		int zMax = mini2(int((triMax.z - min.z) / voxel.z) + 1, divZ - 1);

		for (int z = zMin; z <= zMax; ++z)
		{
			for (int y = yMin; y <= yMax; ++y)
			{
				for (int x = xMin; x <= xMax; ++x)
				{
					float3 cellMin = make_float3
					(
						min.x + voxel.x * x,
						min.y + voxel.y * y,
						min.z + voxel.z * z
					);
					float3 cellMax = cellMin + voxel;
					
					float3 center      = (cellMax + cellMin) / 2;
					float3 halfsize    = (cellMax - cellMin) / 2;
					float _center[3]   = {center.x,   center.y,   center.z};
					float _halfsize[3] = {halfsize.x, halfsize.y, halfsize.z};
					
					float4 v0 = tri.v0;
					float4 v1 = tri.v0 + tri.e1;
					float4 v2 = tri.v0 + tri.e2;
					
					float triverts[3][3] = {
						{v0.x, v0.y, v0.z},
						{v1.x, v1.y, v1.z},
						{v2.x, v2.y, v2.z}
					};
					
					int goes_in = triBoxOverlap(_center,_halfsize,triverts);

					// if (triBoxTest(tri, cellMin, cellMax))
					if (goes_in)
					{
						Node *& node = at(x, y, z);
						if (!node)
						{
							node = new Cell;
						}
						if (node->type() == CELL)
						{
							dynamic_cast<Cell *>(node)->indexes.push_back(index);
						}
						else
						{
							dynamic_cast<Grid *>(node)->add(tri, index, cellMin, cellMax);
						}
					}
				}
			}
		}
	}

	void extend(const Tri *tris, int level, int maxLevels, const float3& alpha, const float3& min, const float3& max)
	{
		if (level == maxLevels - 1)
		{
			return;
		}

		float3 size = max - min;
		float3 voxel = make_float3(size.x / divX, size.y / divY, size.z / divZ);

		for (int z = 0; z < divZ; ++z)
		{
			for (int y = 0; y < divY; ++y)
			{
				for (int x = 0; x < divX; ++x)
				{
					if (at(x, y, z) && at(x, y, z)->type() == CELL
					 && dynamic_cast<Cell *>(at(x, y, z))->indexes.size() >= THRESHOLD)
					{
						float3 cellMin = make_float3
						(
							min.x + voxel.x * x,
							min.y + voxel.y * y,
							min.z + voxel.z * z
						);
						float3 cellMax = cellMin + voxel;

						Cell *cell = dynamic_cast<Cell *>(at(x, y, z));

						int n = cell->indexes.size();
						float n13 = powf(n, 1.0f / 3.0f);
						int divX = (int) (alpha.x * n13);
						int divY = (int) (alpha.y * n13);
						int divZ = (int) (alpha.z * n13);

						Grid *grid = new Grid(divX, divY, divZ);
						std::vector<int>::const_iterator i;
						for (i = cell->indexes.begin(); i != cell->indexes.end(); ++i)
						{
							grid->add(tris[*i], *i, cellMin, cellMax);
						}
						grid->extend(tris, level + 1, maxLevels, alpha, cellMin, cellMax);

						at(x, y, z) = grid;
						delete cell;
					}
				}
			}
		}
	}
};

#endif // GRID_H
