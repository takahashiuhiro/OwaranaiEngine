#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{
    bool isGPUDevice = 1;
    Tensor* a = new Tensor({3,2}, isGPUDevice, {1.08,2,3,4,8,9});
    Tensor* b = new Tensor({2,4}, isGPUDevice, {4,3,2,1,-1,-15000,88.1,55.9});
    //a->SumTensorDim(1)->PrintData();
    b->T()->Matmul(a->T()->T()->T()->T()->T())->T()->T()->T()->T()->T()->T()->PrintData();
    b->T()->Matmul(a->T()->T()->T()->T()->T())->T()->T()->T()->T()->T()->T()->Sum({0})->PrintData();
    //a->PrintData();
    //a->T()->T()->T()->T()->T()->T()->T()->T()->T()->T()->T()->T()->PrintData();

    //DynamicTensor dta(std::shared_ptr<Tensor>(a), 1);
    //DynamicTensor dtb(std::shared_ptr<Tensor>(b), 1);
    //print(dta+dtb);
    //a->Add(b)->PrintData();
    //b->PrintData();
    //a->FillArray(50);
    //a->AddArray(b)->PrintData();
    //b->PrintData();
    //auto dtc = (dta%dtb).Sum();
    //print(dtc);
    //dtc.Backward();
    //print(dta.Grad());
    //print(dtb.Grad());

}