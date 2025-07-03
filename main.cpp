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
    //NESGMMBased<yxx> solver;
    //he params = he::NewDict();
    //params["DimNum"] = 10;
    //params["CosmosNum"] = 10;
    //params["SampleNum"] = 10;
    //solver.Init(params);
    //solver.Solve();

    std::vector<DynamicTensor>hj;
    DynamicTensor a = DynamicTensor({1,2}, {1.,2},0,1);
    print(a%a.Transpose(0,1));
}