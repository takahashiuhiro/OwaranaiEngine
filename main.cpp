#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;
    Tensor* q =new Tensor(std::vector<size_t>{2,3,4}, "GPU", 0);
    q->FillArray(18.);
    Tensor* w =new Tensor(std::vector<size_t>{2,3,4}, "GPU", 0);
    w->FillArray(2.);
    q->SetV(std::vector<size_t>{1,0,2}, 99.);
    q->SetV(std::vector<size_t>{0,0,1}, 77.);
    w->SetV(std::vector<size_t>{1,2,2}, 899.);
    //Tensor* e = q->AverageTensorDim(0);
    //q->PrintData();
    //e->PrintData();
    //w->PrintData();
    //e->PrintData();
    CGNode *rq = new CGNode(q, 1);
    CGNode *rw = new CGNode(w, 1);
    CGNode *re = new CGNode(std::vector<CGNode*>{rq,rw},"Add", 1);
    re->Forward();

    rq->NodeContent->PrintData();
    rw->NodeContent->PrintData();
    re->NodeContent->PrintData();
    re->Gradient->PrintData();
    //rf->Gradient->PrintData();

    std::cout<<"------------------------CPU test---------------------------------"<<std::endl;
    Tensor* qq =new Tensor(std::vector<size_t>{2,3,4});
    qq->FillArray(18.);
    Tensor* wq =new Tensor(std::vector<size_t>{2,3,4});
    wq->FillArray(2.);
    qq->SetV(std::vector<size_t>{1,0,2}, 99.);
    qq->SetV(std::vector<size_t>{0,0,1}, 77.);
    wq->SetV(std::vector<size_t>{1,2,2}, 899.);
    //Tensor* eq = qq->AverageTensorDim(0);
    //qq->PrintData();
    //eq->PrintData();
    //wq->PrintData();
    //eq->PrintData();
    CGNode *rqq = new CGNode(qq, 1);
    CGNode *rwq = new CGNode(wq, 1);
    CGNode *req = new CGNode(std::vector<CGNode*>{rqq,rwq},"Add", 1);
    req->Forward();

    rqq->NodeContent->PrintData();
    rwq->NodeContent->PrintData();
    req->NodeContent->PrintData();
    req->Gradient->PrintData();
    //rf->Gradient->PrintData();
}
