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
	DynamicTensor q1({ 2,3 }, 1);
	DynamicTensor q2({ 3,4 }, 1);
	DynamicTensor q3({ 4,5 }, 1);
	DynamicTensor q4({ 5,6 }, 1);

	q1.Fill(1);
	q2.Fill(2);
	q3.Fill(3);
	q4.Fill(4);

	DynamicTensor e1 = q1%q2%q3%q4;	
	print(e1);
	print(55);
	print(-66);
	print("asdasd");

	e1.Backward();

	print(q1.GetGrad());
}
