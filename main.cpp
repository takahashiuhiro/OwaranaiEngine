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
	DynamicTensor q1({1,3,1,2}, 1,1);
	DynamicTensor q2({3,1,2,1}, 1,1);


	q1.Fill(1);
	q2.Fill(2);

	DynamicTensor e1;
	e1 = q2%q1;

	//e1 = e1.Sum({ 1,3 });
	//e1 = e1.View({1,-1});
	e1 = e1.Softmax(1);
	
	Tensor* loss = new Tensor({3,3,2,2},1);
	loss->FillArray(1.);
	loss->SetV({1,2,1,0},55.);
	loss->SetV({0,2,1,0},345.);
	loss->SetV({0,1,0,1},95.);
	loss->SetV({2,0,0,0},1082.);

	DynamicTensor loss1 = DynamicTensor(std::shared_ptr<Tensor>(loss));

	e1.Backward(loss1);
	print(e1);
	e1.Ops->TensorPointer->PrintShape();
	print(q1.GetGrad());
	print(q2.GetGrad());
}
