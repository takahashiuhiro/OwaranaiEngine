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
	params["NEmbd"] = 6;
	params["Dropout"] = 0.;
	params["BlockSize"] = 40;
	params["Bias"] = 0;

	BaseDynamicLayer* ly = new CausalSelfAttention();
	ly->Init(params);
	ly->SubLayers["CAttn"]->Weights["Weight"].Fill(1);
	ly->SubLayers["CProj"]->Weights["Weight"].Fill(1.5);

	DynamicTensor x({4,5,6},1,0);
	x.Fill(1);

	auto res = ly->Forward({x})[0].Sum();

	//DynamicTensor xx({4,5,6},0,0);
	//xx.Fill(1);
	//print(res);
	res.Backward();

	print(res);
	print(x.Grad());
}
