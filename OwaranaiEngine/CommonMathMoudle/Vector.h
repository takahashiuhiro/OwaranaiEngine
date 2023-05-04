#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"
//#include "Tensor.h"

struct Vector
{
public:

	Vector()
	{
		Data = new Tensor(std::vector<size_t>{1,3}, "CPU", 0);
	}
	Vector(double X, double Y)
	{
		Data = new Tensor(std::vector<size_t>{1,3}, "CPU", 0);
		Data->DataCPU[0] = X;
		Data->DataCPU[1] = Y;
		Data->DataCPU[2] = 0;
	}
	Vector(double X, double Y, double Z)
	{
		Data = new Tensor(std::vector<size_t>{1,3}, "CPU", 0);
		Data->DataCPU[0] = X;
		Data->DataCPU[1] = Y;
		Data->DataCPU[2] = Z;
	}
	~Vector(){};

	Tensor* Data; 

	double X(){return Data->DataCPU[0];}
	double Y(){return Data->DataCPU[1];}
	double Z(){return Data->DataCPU[2];}
	void SetX(double Value){Data->DataCPU[0] = Value;}
	void SetY(double Value){Data->DataCPU[1] = Value;}
	void SetZ(double Value){Data->DataCPU[2] = Value;}

	/**重载基本运算.*/
	Vector operator + (const Vector &Right) { return Vector(Data->DataCPU[0] + Right.Data->DataCPU[0], Data->DataCPU[1] + Right.Data->DataCPU[1], Data->DataCPU[2] + Right.Data->DataCPU[2]); }
	Vector operator - (const Vector &Right) { return Vector(Data->DataCPU[0] - Right.Data->DataCPU[0], Data->DataCPU[1] - Right.Data->DataCPU[1], Data->DataCPU[2] - Right.Data->DataCPU[2]); }
	Vector operator * (const double &Right) { return Vector(Data->DataCPU[0] * Right, Data->DataCPU[1] * Right, Data->DataCPU[2] * Right); }
	double operator * (const Vector &Right) { return (Data->DataCPU[0] * Right.Data->DataCPU[0]) + (Data->DataCPU[1] * Right.Data->DataCPU[1]) + (Data->DataCPU[2] * Right.Data->DataCPU[2]); }
	Vector operator / (const double &Right) { return Vector(Data->DataCPU[0] / Right, Data->DataCPU[1] / Right, Data->DataCPU[2]/Right); }

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
	void PrintData() {std::cout << "X():" << X() << " Y():" << Y() << " Z():"<<Z() << std::endl;}
};