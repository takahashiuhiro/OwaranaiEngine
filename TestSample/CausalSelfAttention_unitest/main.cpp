#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
	he params = he::NewDict();
	params["NHead"] = 3;
	params["NEmbd"] = 3;
	params["Dropout"] = 0.;
	params["BlockSize"] = 40;
	params["Bias"] = 1;

	BaseDynamicLayer* ly = new CausalSelfAttention();
	ly->Init(params);
	ly->SubLayers["CAttn"]->Weights["Weight"].Fill(1);
	ly->SubLayers["CAttn"]->Weights["Weight"].Ops->TensorPointer->SetV({1,2},987);
	ly->SubLayers["CAttn"]->Weights["Bias"].Fill(0.3);
	ly->SubLayers["CProj"]->Weights["Weight"].Fill(1.5);
	ly->SubLayers["CProj"]->Weights["Bias"].Fill(2);

	std::vector<DynamicTensor>inputx = {
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, 0, {1,2,1,6,5,6})), 1),
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, 0, {3,2,3,4,5,4})), 1)
	};

	std::vector<DynamicTensor>outputx = {
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, 0, {1,0,0,0,0,0})), 0),
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, 0, {0,0,0,0,0,1})), 0)
	};

	auto sgd = Optimizer::CreateSGD(ly->Parameters(), 0.01);

	for(int a=0;a<inputx.size();a++)
	{
		auto res = ly->Forward({inputx[a]})[0];
		auto lossres = (res-outputx[a]).Pow(2).Sum()*(1./res.Ops->TensorPointer->ShapeCount);
		lossres.Backward();
		sgd.Step();
		print(res);
		print(lossres);
	}
}
