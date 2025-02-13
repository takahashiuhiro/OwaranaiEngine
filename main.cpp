#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"


int main() 
{
    
    bool isGPUDevice = 1;

    print(DynamicTensor::CreateUnitTensor({2,2},0, isGPUDevice));



    //print(DynamicTensor::)

    //print(DynamicTensor::Arange(5,106,5,0,isGPUDevice));
    //a->SumTensorDim(1)->PrintData();
    //b->T()->Matmul(a->T()->T()->T()->T()->T())->T()->T()->T()->T()->T()->T()->PrintData();
    //b->T()->Matmul(a->T()->T()->T()->T()->T())->T()->T()->T()->T()->T()->T()->Sum({0})->PrintData();
    //a->T()->T()->T()->T()->Maximum({0})->PrintData();
    //a->PrintData();
    //a->T()->T()->T()->T()->T()->T()->T()->T()->T()->T()->T()->T()->PrintData();
    //a->EleInverse()->PrintData();
    //b->GenerateSignTensor()->PrintData();
    //DynamicTensor dta(std::shared_ptr<Tensor>(b), 1);
    //print(dta.EleLog());
    //print(dta.Transpose(0,1));
    //Sb->Sin()->PrintData();
    //auto ff = DynamicTensor::CreateOnehotTensor({1,9}, {1,2,3,5,4,2,0,0,2}, 7, 0, isGPUDevice);
    //print(ff);
    //print(dta.Pow(3.2));
    //DynamicTensor dtb(std::shared_ptr<Tensor>(b), 1);
    //dtb.FillRandomValNormal(0,10);
    //print(dtb);
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