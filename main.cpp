#include "Header/OEDynamic.h"
#include "Header/Application/GPTX/GPTX.h"

struct yxx
{

    OwaranaiEngine::DynamicTensor Forward(OwaranaiEngine::DynamicTensor x)
    {
        // 输出模型的结果
        return x.Sum({2}, true);
    }

    OwaranaiEngine::DynamicTensor Eval(OwaranaiEngine::DynamicTensor x)
    {
        OwaranaiEngine::DynamicTensor ForwardRes = Forward(x);
        double TrueRes = 15;
        OwaranaiEngine::DynamicTensor cost = (ForwardRes + TrueRes*(-1)).Abs()*(-1); // cost计算
        return cost.Sum({2}, true);
    }
};


int main() 
{
    OwaranaiEngine::DynamicTensor s = OwaranaiEngine::DynamicTensor({2,2},{1,2,3,4.},1,1);
    OwaranaiEngine::DynamicTensor d = OwaranaiEngine::DynamicTensor({2,2},{1,2,3,4.},1,1);
    print(s+d);
    /*
    OwaranaiEngine::NESGMMBased<yxx> solver;
    yxx test;
    solver.SetTargetObject(&test);
    OwaranaiEngine::he params = OwaranaiEngine::he::NewDict();
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
    */
}