#pragma once

#include "geometry_include.h"
#include "vector.h"
#include "segment.h"
#include "polygon_2d.h"
#include "convex_polygon_2d.h"


struct plane
{
public:
	plane();
	plane(vector fixed_vector, vector normal_vector);
	plane(vector fixed_vector, vector unit_vector[2]);

	vector fixed_vector;
	vector normal_vector;
	vector unit_vector[2];

	/**the distance from the vector to this plane*/
	double vector_to_plane_dis(vector vec);

	/**the projection from the vector to this plane*/
	vector projection(vector moto_vector);

	/**convert coordinate to this plane by unit vector*/
	vector coordinate_convert_to_plane(vector moto_vector);

	/**convert coordinate revert this plane to 3d space by unit vector*/
	vector coordinate_convert_to_std_space(vector plane_vector);

	/**
	*if two segment cross, cross_vector get a cross vector and the function return True else return false use relative coordinate
	*@param seg_1 a segment
	*@param seg_2 a segment
	*@param cross_vector a cross result by 2 segments cross
	*/
	static bool segment_cross(segment seg_1, segment seg_2, vector& cross_vector);


	/**
	* get a 2d convex polygon from a vector set
	*/
	static bool get_convex_polygon_2d(const std::vector<vector>* vector_set_input, convex_polygon_2d* result_polygon);
}; 