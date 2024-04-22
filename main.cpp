#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"

DynamicTensor rt(DynamicTensor q1, DynamicTensor q2,DynamicTensor q3,DynamicTensor q4)
{
	return q1%q2%q3%q4;
}

int main() 
{
	DynamicTensor q1({ 2,3 }, 1);
	DynamicTensor q2({ 3,4 }, 1);
	DynamicTensor q3({ 4,5 }, 1);
	DynamicTensor q4({ 5,6 }, 1);

	q1.Fill(1);
	q2.Fill(2);
	q3.Fill(3);
	q4.Fill(4);

	DynamicTensor e1 = rt(q1,q2,q3,q4);

	e1.PrintData();

	e1.Backward();

	q1.Ops->GradOps->TensorPointer->PrintData();
	q2.Ops->GradOps->TensorPointer->PrintData();
	q3.Ops->GradOps->TensorPointer->PrintData();
	q4.Ops->GradOps->TensorPointer->PrintData();
}
