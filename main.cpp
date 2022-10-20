#include "TensorCore/Tensor.h"


int main()
{
    Tensor* q =new Tensor(std::vector<size_t>{1,2,3}, "GPU");
    q->FillArray(1.);
    Tensor* w =new Tensor(std::vector<size_t>{1,2,3}, "GPU");
    w->FillArray(2.);
    Tensor* e = q->AddArray(w);
    e->PrintData();
    Tensor* r = e->AddArray(e);
    r->PrintData();
}
