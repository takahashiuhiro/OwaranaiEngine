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
	he lnp = he::NewDict();
	lnp["NormalizedShape"] = he::NewList<int>({ 2,4 });
	BaseDynamicLayer* nm = new LayerNorm();
	nm->Init(lnp);
    auto ppp = nm->Forward({ q })[0];
    print(ppp);
    auto r = ppp.Sum();
    r.Backward();
	print(r);
    print(q.Grad());
}
