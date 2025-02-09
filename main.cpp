#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

//python 语言大师第2课
int main() 
{
    
    bool isGPUDevice = 1;
    DynamicTensor c = DynamicTensor({4,2},{100,200,300,400,500,600,700,800.},1,isGPUDevice);
    DynamicTensor d = DynamicTensor({3,4},{1,2,3,4,5,6,7,8,9,10,11,12.},1,isGPUDevice);
    //c.Fill(5);
    //d.Fill(9);
    he param = he::NewDict();
    std::vector<int>tg = {8,8};
    param["TargetShape"] = he::NewList(tg);
    param["InputStartShape"] = he::NewList(2);
    param["SubInputShapeS"] = he::NewList(2);
    param["SubInputShapeE"] = he::NewList(2);
    tg = {1,1};
    param["InputStartShape"][0] = he::NewList(tg);
    tg = {0,0};
    param["SubInputShapeS"][0] = he::NewList(tg);
    tg = {3,1};
    param["SubInputShapeE"][0] = he::NewList(tg);
    tg = {3,3};
    param["InputStartShape"][1] = he::NewList(tg);
    tg = {1,0};
    param["SubInputShapeS"][1] = he::NewList(tg);
    tg = {2,3};
    param["SubInputShapeE"][1] = he::NewList(tg);

    auto ggg = DynamicTensor::DynamicStdOps_Forward_SubSend({c,d},param,1);
    auto pp = ggg.Sum();
    pp.Backward();
    print(ggg);
    print("");

    print(c.Grad());


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