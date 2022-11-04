#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;
    //todo 这种情况下不知道为什么只有batch为0的那一个矩阵生效了
    Tensor* q =new Tensor(std::vector<size_t>{3,4}, "GPU", 0);
    q->FillArray(18.);
    Tensor* w =new Tensor(std::vector<size_t>{4}, "GPU", 0);
    w->FillArray(2.);
    q->SetV(std::vector<size_t>{1,2}, 99.);
    w->SetV(std::vector<size_t>{2}, 899.);
    Tensor* e = q->Matmul(w);
    q->PrintData();
    w->PrintData();
    e->PrintData();
    std::cout<<"------------------------CPU test---------------------------------"<<std::endl;
    Tensor* qq =new Tensor(std::vector<size_t>{3,4});
    qq->FillArray(18.);
    Tensor* wq =new Tensor(std::vector<size_t>{4});
    wq->FillArray(2.);
    qq->SetV(std::vector<size_t>{1,2}, 99.);
    wq->SetV(std::vector<size_t>{2}, 899.);
    Tensor* eq = qq->Matmul(wq);
    qq->PrintData();
    wq->PrintData();
    eq->PrintData();
}
