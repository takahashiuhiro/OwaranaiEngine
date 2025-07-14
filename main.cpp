#include "Code/OEDynamic.h"
#include <chrono>

struct yxx
{

    DynamicTensor Forward(DynamicTensor x)
    {
        // 输出模型的结果
        return x*10;
    }

    DynamicTensor Eval(DynamicTensor x)
    {
        DynamicTensor ForwardRes = Forward(x);
        double TrueRes = 144;
        DynamicTensor cost = (ForwardRes + TrueRes*(-1)).Abs(); // cost计算
        return cost.Sum({1}, true);
    }
};


int main() 
{
    DynamicTensor gg = DynamicTensor({2,3,3}, {4,1,1,1,3,0,1,0,2,2,-1,0,-1,2,-1,0,-1,2.});
    auto ss = gg.Cholesky();
    print(gg);
    print(ss);
    print(ss%ss.Transpose(1,2));
    print(gg.Det_Symmetric(ss));

    return 0;
    NESGMMBased<yxx> solver;
    he params = he::NewDict();
    params["DimNum"] = 3;
    params["CosmosNum"] = 1;
    params["SampleNum"] = 10;
    solver.Init(params);
    solver.Solve();
}