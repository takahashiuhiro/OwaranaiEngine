#pragma once

#include "vector.h"
#include "geometry_include.h"

struct plane;

struct polygon_2d
{
public:
	polygon_2d();
	polygon_2d(std::vector<vector>* vector_list);
	~polygon_2d();

	/*point set*/
	std::vector<vector>* vector_list;


};

