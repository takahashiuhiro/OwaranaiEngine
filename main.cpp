#include "Code/OEDynamic.h"
#include <chrono>

struct yxx
{

    DynamicTensor Forward(DynamicTensor x)
    {
        // 输出模型的结果
        return x.Sum({2}, true);
    }

    DynamicTensor Eval(DynamicTensor x)
    {
        DynamicTensor ForwardRes = Forward(x);
        double TrueRes = 15;
        DynamicTensor cost = (ForwardRes + TrueRes*(-1)).Abs()*(-1); // cost计算
        return cost.Sum({2}, true);
    }
};

int main() 
{
    NESGMMBased<yxx> solver;
    he params = he::NewDict();
    params["DimNum"] = 3;
    params["CosmosNum"] = 2;
    params["SampleNum"] = 20;
    params["MaxItNum"] = 150;
    params["HistoryLength"] = 2;
    params["LearingRate_Mean"] = 0.3;
    params["LearingRate_Var"] = 0.2;
    params["Beta"] = 0.4;
    solver.Init(params);
    print(solver.Solve());
}