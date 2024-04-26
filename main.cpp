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
	DynamicTensor q1({1,3,1,2}, 1);
	DynamicTensor q2({3,1,2,1}, 1);

	q1.Fill(1);
	q2.Fill(2);

	DynamicTensor e1 = q1 + q2;

	e1 = e1.Sum({ 1,3 });
	e1 = e1.View({1,-1});
	print(e1);
	e1.Ops->TensorPointer->PrintShape();
	
	
	e1.Backward();
	
	print(q1.GetGrad());
	print(q2.GetGrad());
}
