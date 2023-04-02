#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 
    Tensor* qee = new Tensor(std::vector<size_t>{1,3,4});
    float* ew = new float[12]{1,2,3,4,5,6,7,8,9,11,12,13};
    for(int a=0;a<12;a++)
    {
        qee->DataCPU[a] = ew[a];
    }
    //qee->ToGPU();
    qee->GaussianElimination();
    qee->PrintData();
}
