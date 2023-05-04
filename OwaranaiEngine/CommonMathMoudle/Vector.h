#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"

struct Vector:Tensor
{
public:

	Vector()
	{
		this->shape = std::vector<size_t>{1,3};
    	ShapeCount = 3;
    	this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
	}
	Vector(double X, double Y)
	{
		this->shape = std::vector<size_t>{1,3};
    	ShapeCount = 3;
    	this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
		DataCPU[0] = X;
		DataCPU[1] = Y;
		DataCPU[2] = 0;
	}
	Vector(double X, double Y, double Z)
	{
		this->shape = std::vector<size_t>{1,3};
    	ShapeCount = 3;
    	this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
		DataCPU[0] = X;
		DataCPU[1] = Y;
		DataCPU[2] = Z;
	}
	~Vector(){}

	double X(){return DataCPU[0];}
	double Y(){return DataCPU[1];}
	double Z(){return DataCPU[2];}
	void SetX(double Value){DataCPU[0] = Value;}
	void SetY(double Value){DataCPU[1] = Value;}
	void SetZ(double Value){DataCPU[2] = Value;}

	/**重载基本运算.*/
	Vector operator + (const Vector &Right) { return Vector(DataCPU[0] + Right.DataCPU[0], DataCPU[1] + Right.DataCPU[1], DataCPU[2] + Right.DataCPU[2]); }
	Vector operator - (const Vector &Right) { return Vector(DataCPU[0] - Right.DataCPU[0], DataCPU[1] - Right.DataCPU[1], DataCPU[2] - Right.DataCPU[2]); }
	Vector operator * (const double &Right) { return Vector(DataCPU[0] * Right, DataCPU[1] * Right, DataCPU[2] * Right); }
	double operator * (const Vector &Right) { return (DataCPU[0] * Right.DataCPU[0]) + (DataCPU[1] * Right.DataCPU[1]) + (DataCPU[2] * Right.DataCPU[2]); }
	Vector operator / (const double &Right) { return Vector(DataCPU[0] / Right, DataCPU[1] / Right, DataCPU[2]/Right); }

	/**向量长度.*/
	double Length() { return sqrt(X() * X() + Y() * Y() + Z() * Z()); }

	/**单位向量.*/
	Vector DirectionVector()
	{
		double VectorLength = Length();
		if (VectorLength * VectorLength < 1e-8) return Vector(0, 0);
		return *this / VectorLength;
	}

	/**向量叉乘.*/
	Vector CrossProduct(Vector Right) { return Vector(Y() * Right.Z() - Right.Y() * Z(), Right.X() * Z() - X() * Right.Z(), X() * Right.Y() - Right.X() * Y()); }

	/**打印数据.*/
	void PrintData() {std::cout << "X:" << X() << " Y:" << Y() << " Z:"<<Z() << std::endl;}
};