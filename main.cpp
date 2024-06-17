#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
	DynamicTensor q(std::shared_ptr<Tensor>(new Tensor({ 3, 2, 4 }, 0, { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24. })),true);
	he MLPparams = he::NewDict();
	MLPparams["InChannels"] = 4;
	MLPparams["HiddenChannels"] = he::NewList<int>({ 4,6,8,2 });
	MLPparams["NormLayer"] = "LayerNorm";
	MLPparams["ActivationLayer"] = "GELU";
	BaseDynamicLayer* mlp = new MLP();
	mlp->Init(MLPparams);
    auto ppp = mlp->Forward({ q })[0];
    print(ppp);
    auto r = ppp.Sum();
    r.Backward();
	print(r);
    print(q.Grad());
}
