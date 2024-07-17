#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
	size_t dvnum = 0;


	he params = he::NewDict();
	params["NHead"] = 3;
	params["NEmbd"] = 3;
	params["Dropout"] = 0.;
	params["BlockSize"] = 40;
	params["Bias"] = 1;
	params["DeviceNum"] = int(dvnum);
	params["InChannels"] = 36;
	params["HiddenChannels"] = he::NewList<int>({1080,1080,10});

	BaseDynamicLayer* ly = new MLP();
	ly->Init(params);

	std::vector<float> inputttt = {
    1, 0, 0, 0, 0, 1,  // 数字0
    1, 0, 0, 0, 1, 0,
    1, 0, 0, 1, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1,
    
    0, 1, 0, 0, 0, 0,  // 数字1
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    
    1, 1, 1, 1, 1, 1,  // 数字2
    0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    
    1, 1, 1, 1, 1, 1,  // 数字3
    0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    
    1, 0, 0, 1, 0, 0,  // 数字4
    1, 0, 0, 1, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0,
    
    1, 1, 1, 1, 1, 1,  // 数字5
    1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    
    1, 1, 1, 1, 1, 1,  // 数字6
    1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 1, 0, 0,
    1, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 0,
    
    1, 1, 1, 1, 1, 1,  // 数字7
    0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    
    1, 1, 1, 1, 1, 1,  // 数字8
    1, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1,
    
    1, 1, 1, 1, 1, 1,  // 数字9
    1, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1
};

std::vector<float> outputttt = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 数字0
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  // 数字1
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  // 数字2
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // 数字3
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  // 数字4
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  // 数字5
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  // 数字6
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  // 数字7
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // 数字8
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1   // 数字9
};


	std::vector<DynamicTensor>inputx = {
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({10,params["InChannels"].i()}, dvnum, inputttt)), 1),
	};

	std::vector<DynamicTensor>outputx = {
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({10,10}, dvnum, outputttt)), 0),
	};

	auto sgd = Optimizer::CreateSGD(ly->Parameters(), 0.01);

	for(int a=0;a<inputx.size()+5000;a++)
	{
		auto res = ly->Forward({inputx[0]})[0].Softmax(-1);
		auto lossres = (res-outputx[0]).Pow(2).Sum()*(1./res.Ops->TensorPointer->ShapeCount);
		lossres.Backward();
		sgd.Step();
		//print(res);
		if(a == inputx.size()+4999)
		{print(lossres);
		print("");
		}
	}
	print(ly->Forward({inputx[0]})[0].Softmax(-1));
}
