#include "Code/OEDynamic.h"

int main() 
{
    //auto q = new GPT2Model();
    //he Params = he::NewDict();
    //Params["BlockSize"] = 1;
    //Params["VocabSize"] = 1;
    //Params["NLayers"] = 1;
    //Params["NHead"] = 1;
    //Params["NEmbd"] = 1;
    //Params["Dropout"] = float(1.);
    //Params["Bias"] = 1;
    //q->Init(Params);
    //print(q->GetNumParams());
    std::vector<float>dt = {1,2,3,4,5,M_E,7,8,M_PI, M_PI*2, M_E*3, M_E*4};
    std::vector<size_t>dts = {2,2,3};
    int dvm = 0;

    DynamicTensor qq(std::shared_ptr<Tensor>(new Tensor(dts, dvm, dt)), 1);
    DynamicTensor ww(std::shared_ptr<Tensor>(new Tensor(dts, dvm, dt)), 0);

    auto g = DynamicTensor::CrossEntropy(qq,ww,"Sum");

    g.Backward();

    print(qq);
    print(g);
    print(qq.Grad());
}