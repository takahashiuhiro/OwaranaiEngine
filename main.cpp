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

	DynamicTensor ss({3,3},1);
	ss.Fill(1);
	ss.Ops->IsEval = true;
	auto gg = DynamicTensor::Dropout(ss,0.4);
	print(gg);
	return 0;

	DynamicTensor q1({1,3,1}, 1,0);
	DynamicTensor q2({3,1,1}, 1,0);


	q1.Fill(1);
	q2.Fill(2);


	DynamicTensor e1;

	e1 = q1+q2;
	//e1 = q1*q2;

	//e1 = e1.Sum({ 1,3 });
	//e1 = e1.View({1,-1});
	e1 = e1.Softmax(1);
	e1 = e1.Pow(3);
	Tensor* loss = new Tensor({3,3,1},0);
	loss->FillArray(1.);
	loss->SetV({1,2,0},55.);
	loss->SetV({0,2,0},345.);
	loss->SetV({1,0,0},753.);
	loss->SetV({1,1,0},753.);
	loss->SetV({2,1,0},999.);
	//loss->SetV({0,1,0,0},95.);
	//loss->SetV({2,0,0,0},1082.);

	DynamicTensor loss1 = DynamicTensor(std::shared_ptr<Tensor>(loss));

	e1.Backward(loss1);
	//print(e1);
	//e1.Ops->TensorPointer->PrintShape();
	print(q1.GetGrad());
	print("---final_print-----");
	print(q2.GetGrad());
}
