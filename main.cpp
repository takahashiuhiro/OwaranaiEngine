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


    DynamicTensor tt = DynamicTensor::SampleFromStdGaussian(2, {2,3}, -1, 0);
    print(tt);
}