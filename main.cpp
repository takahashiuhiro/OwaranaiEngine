#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"

DynamicTensor rtrt(DynamicTensor f1, DynamicTensor f2, DynamicTensor f3, DynamicTensor f4, DynamicTensor f5, DynamicTensor f6)
{
	DynamicTensor q1 = f1;
	DynamicTensor q2 = f2;
	DynamicTensor q3 = f3;
	DynamicTensor q4 = f4;
	DynamicTensor q5 = f5;
	DynamicTensor q6 = f6;
	DynamicTensor e1 = q1 + q2;


	DynamicTensor e7 = q1+ e1 + q3 + q4 + q5;
	DynamicTensor e8 = q6 + e1 + q3 + q4 + q5;
	DynamicTensor e9 = DynamicTensor::DynamicStdOps_Forward_Add({ e7,e8 }, he(), 1);
	DynamicTensor e10 = DynamicTensor::DynamicStdOps_Forward_Add({ e9,q6 }, he(), 1);
	DynamicTensor e11 = DynamicTensor::DynamicStdOps_Forward_Add({ e10,e9 }, he(), 1);
	return e11;
}

int main() 
{
	DynamicTensor q1 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, {2,3.})),1);
	DynamicTensor q2 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 4,7. })),1);
	DynamicTensor q3 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 2,3. })), 1);
	DynamicTensor q4 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 4,7. })), 1);
	DynamicTensor q5 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 2,3. })), 1);
	DynamicTensor q6 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 4,7. })), 1);
	
	
	DynamicTensor e11 = rtrt(q1, q2, q3, q4, q5, q6);

	DynamicTensor q66 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 99,100. })), 1);

	e11.Backward(&q66);
	q1.Ops->GradOps->TensorPointer->PrintData();
	e11.Ops->TensorPointer->PrintData();

	(q1 + q2).Ops->TensorPointer->PrintData();
}
