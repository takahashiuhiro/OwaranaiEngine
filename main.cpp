#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;
    //todo 这种情况下不知道为什么只有batch为0的那一个矩阵生效了
    Tensor* q =new Tensor(std::vector<size_t>{1,4,5,3}, "GPU", 0);
    q->FillArray(18.);
    Tensor* w =new Tensor(std::vector<size_t>{8,1,3,7}, "GPU", 0);
    w->FillArray(2.);
    q->SetV(std::vector<size_t>{0,2,1,2}, 99.);
    w->SetV(std::vector<size_t>{5,0,2,4}, 899.);
    Tensor* e = q->Matmul(w);
    q->PrintData();
    w->PrintData();
    e->PrintData();
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
