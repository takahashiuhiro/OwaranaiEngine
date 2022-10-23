#pragma once

#include "vector.h"

struct segment
{
public:
	
	segment();
	vector start_vector;
	vector end_vector;
	vector direction_vector;

	/**segment init by head and tail vector*/
	segment init(vector start_vector, vector end_vector);

	/**the length of this segment*/
	double length();

	/**the projection from vector to this segment*/
	vector projection(vector moto_vector);

};