#include "TensorCore/Tensor.h"


int main()
{
    std::cout<<"gpu test"<<std::endl;
    Tensor* q =new Tensor(std::vector<size_t>{1,2,3}, "GPU", 0);
    q->FillArray(1.);
    Tensor* w =new Tensor(std::vector<size_t>{1,2,3}, "GPU", 0);
    w->FillArray(2.);
    Tensor* e = q->AddArray(w);
    e->PrintData();
    Tensor* r = e->AddArray(e);
    r->PrintData();
    std::cout<<"cpu test"<<std::endl;
    Tensor* qq =new Tensor(std::vector<size_t>{3,2,2,3});
    qq->FillArray(1.);
    Tensor* ww =new Tensor(std::vector<size_t>{2,2,3});
    ww->FillArray(0.);
    ww->SetV(std::vector<size_t>{0,0,0}, 5);
    //std::cout<<ww->GetV(std::vector<size_t>{0,0,0})<<std::endl;
    Tensor* ee = qq->Add(ww);
    ee->PrintData();
    //Tensor* rr = ee->AddArray(ee);
    //rr->PrintData();
}
