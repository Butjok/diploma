#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <string.h>

#ifndef __CUDACC__
#	define CPU 1
#	define GPU 0
#else
#	define CPU 0
#	define GPU 1
#endif

#include <math.h>
#include <vector_types.h>

#define USE_GRID 1

#endif // COMMON_H
