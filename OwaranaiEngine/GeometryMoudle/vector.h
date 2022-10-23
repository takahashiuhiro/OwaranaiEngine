#pragma once

#include"node.h"
#include<math.h>

struct vector :node
{
public:
	vector();
	vector(double x, double y);
	vector(double x, double y, double z);
	~vector();

	double x, y, z;

	vector operator + (const vector& right);
	vector operator - (const vector& right);
	vector operator * (const double& right);
	double operator * (const vector& right);
	vector operator / (const double& right);

	double length();
	vector direction_vector();
	vector cross_product(vector right);

	void print();

	/**compare with x and y*/
	static bool vector_compare_x_up(vector a, vector b);
};