#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{
    bool isGPUDevice = 1;
    Tensor* a = new Tensor({3,2}, isGPUDevice, {1,2,3,4,8,9});
    Tensor* b = new Tensor({2,3}, isGPUDevice, {4,3,2,1,-1,-15000});
    //a->Matmul(b)->PrintData();
    Tensor* aa = new Tensor({4,30000}, isGPUDevice);
    Tensor* bb = new Tensor({30000,7}, isGPUDevice);
    aa->FillArray(-102);
    bb->FillArray(-37);
    aa->Matmul(bb)->PrintData();

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