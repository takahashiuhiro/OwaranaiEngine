#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicLayers/Linear.h"
#include "Code/DynamicAutomaticDifferentiation/Optimizer.h"

int main() 
{
	DynamicTensor x({ 3,3,5 }, 1);
	x.Fill(3.4);

	Linear* layer = new Linear();
	he params = he::NewDict();
	params["DeviceNum"] = 0;
	params["InFeatures"] = 5;
	params["OutFeatures"] = 4;
	params["Bias"] = 1;

	layer->Init(params);
	auto ty = layer->StateDict();

	auto sgd = Optimizer::CreateSGD(layer->Parameters(),10);

	for (int a = 1; a < 3; a++)
	{
		sgd.ZeroGrad();
		auto gg = layer->Forward({ x })[0];
		gg.Backward();
		print("bencikaishi::");
		for (auto& it : ty)
		{
			print(it.first);
			print(it.second.GetGrad());
			print(it.second);
		}
		sgd.Step();
		for (auto& it : ty)
		{
			print(it.first);
			print(it.second.GetGrad());
			print(it.second);
		}
		print("bencijieshu::");
	}

}
