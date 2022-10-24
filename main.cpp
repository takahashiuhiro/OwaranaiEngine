#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;
    Tensor* q =new Tensor(std::vector<size_t>{}, "GPU", 0);
    q->FillArray(10.);
    Tensor* w =new Tensor(std::vector<size_t>{}, "GPU", 0);
    w->FillArray(2.);
    w->SetV(std::vector<size_t>{}, 99.);
    //w->SetV(std::vector<size_t>{0}, 899.);
    Tensor* e = q->AddScalar(8.);
    q->PrintData();
    w->PrintData();
    e->PrintData();
    std::cout<<"------------------------CPU test---------------------------------"<<std::endl;
    Tensor* qq =new Tensor(std::vector<size_t>{});
    qq->FillArray(10.);
    Tensor* wq =new Tensor(std::vector<size_t>{});
    wq->FillArray(2.);
    wq->SetV(std::vector<size_t>{}, 99.);
    //wq->SetV(std::vector<size_t>{0}, 899.);
    Tensor* eq = qq->AddScalar(8.);
    qq->PrintData();
    wq->PrintData();
    eq->PrintData();
}
