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

    int dim = 2;
    DynamicTensor Mean = DynamicTensor({1,1,1,dim}, {10,20.});
    DynamicTensor Var = DynamicTensor({1,1,1,dim,dim}, {100,0,0,1600.});
    std::vector<int>sp = {2,3,4};
    DynamicTensor res = DynamicTensor::SampleFromOtherGaussian(dim, sp, Mean, Var);
    print(res);
}