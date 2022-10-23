#pragma once

#include "geometry_include.h"
#include "vector.h"


struct convex_polygon_2d
{
public:
	convex_polygon_2d();
	convex_polygon_2d(std::vector<vector>* vector_list);
	~convex_polygon_2d();

	/*point set*/
	std::vector<vector>* vector_list;
};

