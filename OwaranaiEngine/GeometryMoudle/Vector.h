#pragma once
#include "StdInclude.h"

struct Vector
{
public:
	Vector(){};
	Vector(double X, double Y)
	{
		this->X = X;
		this->Y = Y;
		this->Z = 0;
	}
	Vector(double X, double Y, double Z)
	{
		this->X = X;
		this->Y = Y;
		this->Z = Z;
	}
	~Vector(){};

	double X, Y, Z;

	/**重载基本运算.*/
	Vector operator + (const Vector& Right) { return Vector(X + Right.X, Y + Right.Y, Z + Right.Z); }
	Vector operator - (const Vector& Right) { return Vector(X - Right.X, Y - Right.Y, Z - Right.Z); }
	Vector operator * (const double& Right) { return Vector(X * Right, Y * Right, Z * Right); }
	double operator * (const Vector& Right) { return (X * Right.X) + (Y * Right.Y) + (Z * Right.Z); }
	Vector operator / (const double& Right) { return Vector(X / Right, Y / Right, Z/Right); }

	/**向量长度.*/
	double Length() { return sqrt(X * X + Y * Y + Z * Z); }

	/**单位向量.*/
	Vector DirectionVector()
	{
		double VectorLength = Length();
		if (VectorLength * VectorLength < 1e-8) return Vector(0, 0);
		return *this / VectorLength;
	}

	/**向量叉乘.*/
	Vector CrossProduct(Vector Right) { return Vector(Y * Right.Z - Right.Y * Z, Right.X * Z - X * Right.Z, X * Right.Y - Right.X * Y); }

	/**打印数据.*/
	void PrintData() {std::cout << "X:" << X << " X:" << Y << " Z:"<<Z << std::endl;}
};