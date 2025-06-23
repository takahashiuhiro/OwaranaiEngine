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
    NESGMMBased<yxx> solver;
    he params = he::NewDict();
    params["DimNum"] = 1;
    params["CosmosNum"] = 5;
    params["SampleNum"] = 30;
    solver.Init(params);
    solver.Solve();

    //std::vector<DynamicTensor>hj;
    //DynamicTensor a = DynamicTensor({1,1}, {-0.5},0,0);
    //hj.push_back(a);
    //hj.back().Abs();
}