#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"


int main() 
{
	DynamicTensor q1 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, {2,3.})),1);
	DynamicTensor q2 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 4,7. })),1);
	DynamicTensor q3 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 2,3. })), 1);
	DynamicTensor q4 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 4,7. })), 1);
	DynamicTensor q5 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 2,3. })), 1);
	DynamicTensor q6 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 4,7. })), 1);
	
	DynamicTensor e1 = DynamicTensor::DynamicStdOps_Forward_Add({ q1,q2 }, he(), 1);
	DynamicTensor e2 = DynamicTensor::DynamicStdOps_Forward_Add({ e1,q3 }, he(), 1);
	DynamicTensor e3 = DynamicTensor::DynamicStdOps_Forward_Add({ e2,q4 }, he(), 1);
	DynamicTensor e4 = DynamicTensor::DynamicStdOps_Forward_Add({ e3,q5 }, he(), 1);

	DynamicTensor e7 = DynamicTensor::DynamicStdOps_Forward_Add({ q1,e4 }, he(), 1);
	DynamicTensor e8 = DynamicTensor::DynamicStdOps_Forward_Add({ e4,q6 }, he(), 1);

	DynamicTensor e9 = DynamicTensor::DynamicStdOps_Forward_Add({ e7,e8 }, he(), 1);
	DynamicTensor e10= DynamicTensor::DynamicStdOps_Forward_Add({ e9,q6 }, he(), 1);
	DynamicTensor e11 = DynamicTensor::DynamicStdOps_Forward_Add({ e10,e9 }, he(), 1);

	DynamicTensor q66 = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 99,100. })), 1);

	e11.Backward(&q66);
	q1.Ops->GradOps->TensorPointer->PrintData();
}
