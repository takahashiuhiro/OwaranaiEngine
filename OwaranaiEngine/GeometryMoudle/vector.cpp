#include"vector.h"

vector::vector(){}
vector::vector(double x, double y)
{
	this->x = x;
	this->y = y;
	this->z = 0;
}
vector::vector(double x, double y, double z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

vector::~vector(){}

vector vector:: operator + (const vector& right) { return vector(x + right.x, y + right.y, z + right.z); }
vector vector:: operator - (const vector& right) { return vector(x - right.x, y - right.y, z - right.z); }
vector vector:: operator * (const double& right) { return vector(x * right, y * right, z * right); }
vector vector:: operator / (const double& right) { return vector(x / right, y / right, z/right); }
double vector:: operator * (const vector& right) { return (x * right.x) + (y * right.y) + (z * right.z); }

double vector::length() { return sqrt(x * x + y * y + z * z); }
vector vector::cross_product(vector right) { return vector(y * right.z - right.y * z, right.x * z - x * right.z, x * right.y - right.x * y); }
vector vector::direction_vector()
{
	double vector_length = length();
	if (vector_length * vector_length < 1e-8) return vector(0, 0);
	return *this / vector_length;
}

void vector::print(){std::cout << "x:" << x << " y:" << y << " z:"<<z << std::endl;}

bool vector::vector_compare_x_up(vector a, vector b)
{
	if ((a.x - b.x) * (a.x - b.x) < 1e-11)return a.y < b.y;
	return a.x < b.x;
}