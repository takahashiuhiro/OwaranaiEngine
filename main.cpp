#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 
    Tensor* qee = new Tensor(std::vector<size_t>{2,3,3});
    float* ew = new float[18]{1,22,3,4,5,6,7,88888,9,11,12,13,10,20,30,40,500,60};
    for(int a=0;a<18;a++)
    {
        qee->DataCPU[a] = ew[a];
    }
    qee->Inverse()->Matmul(qee)->PrintData();
    qee->ToGPU();
    qee->Inverse()->Matmul(qee)->PrintData();
}
