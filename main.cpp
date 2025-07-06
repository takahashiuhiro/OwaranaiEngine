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

    std::vector<float>ff = {
        8.621484, 2.208458, -4.372221, -1.361234, 2.208458, 6.131812, -1.827468, -0.191492, -4.372221, -1.827468, 10.761425, 0.634703, -1.361234, -0.191492, 0.634703, 5.185860,
        6.492416, 0.570125, 1.339495, 1.016229, 0.570125, 5.992822, 0.760973, -0.030761, 1.339495, 0.760973, 5.477684, -0.976499, 1.016229, -0.030761, -0.976499, 7.739824,
    };

    std::vector<DynamicTensor>hj;
    DynamicTensor a = DynamicTensor({2,4,4}, ff,0,0);

    auto gg = a.Ops->TensorPointer->Cholesky();
    gg->PrintData();
    print("--");
    auto ggt = gg->Transpose(1,2);
    gg->Matmul(ggt)->PrintData();
    print(a);
}