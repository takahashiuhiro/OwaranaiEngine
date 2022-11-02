#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;
    Tensor* q =new Tensor(std::vector<size_t>{8,11}, "GPU", 0);
    q->FillArray(18.);
    Tensor* w =new Tensor(std::vector<size_t>{11,9}, "GPU", 0);
    w->FillArray(2.);
    //w->SetV(std::vector<size_t>{}, 99.);
    //w->SetV(std::vector<size_t>{8}, 899.);
    Tensor* e = q->Matmul(w);
    //q->PrintData();
    //w->PrintData();
    //e->PrintData();
    std::cout<<"------------------------CPU test---------------------------------"<<std::endl;
    Tensor* qq =new Tensor(std::vector<size_t>{17});
    qq->FillArray(18.);
    Tensor* wq =new Tensor(std::vector<size_t>{17});
    wq->FillArray(2.);
    //wq->SetV(std::vector<size_t>{}, 99.);
    wq->SetV(std::vector<size_t>{8}, 899.);
    Tensor* eq = qq->Matmul(wq);
    //qq->PrintData();
    //wq->PrintData();
    //eq->PrintData();
}
