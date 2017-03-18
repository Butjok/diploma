#ifndef FLATGRID_H
#define FLATGRID_H

#include "grid.h"

void makeFlatGrid(const Node *node, std::vector<int> *result, const float4 *tris, std::vector<float4> *tris_result, int offset)
{
	if (node->type() == CELL)
	{
		const Cell *cell = dynamic_cast<const Cell *>(node);

		result->push_back(-cell->indexes.size());
		result->push_back(tris_result->size() / 3);
		
		std::vector<int>::const_iterator i;
		for (i = cell->indexes.begin(); i != cell->indexes.end(); ++i)
		{
			// result->push_back(*i);
			int j = *i;
			const float4 v0 = tris[3 * j];
			const float4 e1 = tris[3 * j + 1];
			const float4 e2 = tris[3 * j + 2];
			tris_result->push_back(v0);
			tris_result->push_back(e1);
			tris_result->push_back(e2);
		}
	}
	else
	{
		const Grid *grid = dynamic_cast<const Grid *>(node);

		int size = 3 + grid->divX * grid->divY * grid->divZ;

		std::vector<int> tail;

		result->push_back(grid->divX);
		result->push_back(grid->divY);
		result->push_back(grid->divZ);
		for (int z = 0; z < grid->divZ; ++z)
		{
			for (int y = 0; y < grid->divY; ++y)
			{
				for (int x = 0; x < grid->divX; ++x)
				{
					if (!grid->at(x, y, z))
					{
						result->push_back(-1);
						continue;
					}
					result->push_back(offset + size + tail.size());
					makeFlatGrid(grid->at(x, y, z), &tail, tris, tris_result, offset + size + tail.size());
				}
			}
		}
		std::vector<int>::const_iterator i;
		for (i = tail.begin(); i != tail.end(); ++i)
		{
			result->push_back(*i);
		}
	}
}

#endif // FLATGRID_H
