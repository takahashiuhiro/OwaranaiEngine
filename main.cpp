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
        return cost.Sum({2}, true);
    }
};

int main() 
{

    DynamicTensor gg = DynamicTensor({2,3,2}, {1,2,3,4,5,6,7,8,9,100,11,12.}, 0);
    print(gg);
    print(gg.Max({0,1},true).Shape());
    return 0;
    // test mac
    NESGMMBased<yxx> solver;
    he params = he::NewDict();
    params["DimNum"] = 75;
    params["CosmosNum"] = 2;
    params["SampleNum"] = 4;
    solver.Init(params);
    solver.Solve();
}