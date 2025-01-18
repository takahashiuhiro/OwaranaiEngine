#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{
    bool isGPUDevice = 1;
    Tensor* a = new Tensor({2,2}, isGPUDevice, {1,2,3,4});
    Tensor* b = new Tensor({2,2}, isGPUDevice, {4,3,2,1});
    DynamicTensor dta(std::shared_ptr<Tensor>(a), 1);
    DynamicTensor dtb(std::shared_ptr<Tensor>(b), 1);
    print(dta+dtb);
    //a->Add(b)->PrintData();
    //auto dtc = (dta%dtb).Sum();
    //print(dtc);
    //dtc.Backward();
    //print(dta.Grad());
    //print(dtb.Grad());

}