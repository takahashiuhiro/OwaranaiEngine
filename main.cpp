#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;
    Tensor* loss =new Tensor(std::vector<size_t>{2,1}, "GPU", 0);
    loss->SetV(std::vector<size_t>{0,0}, 100.);
    loss->SetV(std::vector<size_t>{1,0}, 10000.);
    Tensor* q =new Tensor(std::vector<size_t>{2,3}, "GPU", 0);
    q->FillArray(3.);
    Tensor* w =new Tensor(std::vector<size_t>{2,3}, "GPU", 0);
    w->FillArray(5.);
    Tensor* e =new Tensor(std::vector<size_t>{2,3}, "GPU", 0);
    e->FillArray(7.);
    CGNode *rq = new CGNode(q, 1);
    CGNode *rw = new CGNode(w, 1);
    CGNode *re = new CGNode(e, 1);
    CGNode *rf = new CGNode(std::vector<CGNode*>{rq,rw},"Add", 1);
    CGNode *rg = new CGNode(std::vector<CGNode*>{re,rw},"Add", 1);
    CGNode *rh = new CGNode(std::vector<CGNode*>{rf,rw, rg},"Add", 1);
    rh->Forward();
    rh->NodeContent->PrintData();
    rh->Backward(loss);
    rw->DerivativeNode->Forward();
    rw->DerivativeNode->NodeContent->PrintData();
}
