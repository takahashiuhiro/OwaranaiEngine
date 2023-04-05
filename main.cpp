#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 
    Tensor* qee = new Tensor(std::vector<size_t>{2,3,4});
    float* ew = new float[24]{1,2,3,4,5,6,7,8,9,11,12,13,10,20,30,40,50,60,70,80,90,110,120,130};
    for(int a=0;a<24;a++)
    {
        qee->DataCPU[a] = ew[a];
    }
    std::vector<size_t> ss = {1,0,1};
    std::vector<size_t> dd = {1,2,2};
    qee->GetTensorBy2ShapeVector(ss, dd)->PrintData();
    qee->ToGPU();
    qee->GetTensorBy2ShapeVector(ss, dd)->PrintData();
}
