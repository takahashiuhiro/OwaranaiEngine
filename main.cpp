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

    std::cout<<rq<<" q原型\n";
    std::cout<<rw<<" w原型\n";
    std::cout<<re<<" e原型\n";
    std::cout<<rf<<" r原型\n";
    std::cout<<rg<<" g原型\n";
    std::cout<<rh<<" h原型\n";

    std::cout<<rw->DerivativeNode->InputNode.size()<<" 节点数\n";
    //std::cout<<"------------------------CPU test---------------------------------"<<std::endl;
    //Tensor* qq =new Tensor(std::vector<size_t>{2,3,4});
    //qq->FillArray(18.);
    //Tensor* wq =new Tensor(std::vector<size_t>{2,3,4});
    //wq->FillArray(2.);
    //qq->SetV(std::vector<size_t>{1,0,2}, 99.);
    //qq->SetV(std::vector<size_t>{0,0,1}, 77.);
    //wq->SetV(std::vector<size_t>{1,2,2}, 899.);
    //Tensor* lq =new Tensor(std::vector<size_t>{2,1});
    //lq->SetV(std::vector<size_t>{0,0}, 159159.);
    //lq->SetV(std::vector<size_t>{1,0}, 951951.);
    ////Tensor* eq = qq->AverageTensorDim(0);
    ////qq->PrintData();
    ////eq->PrintData();
    ////wq->PrintData();
    ////eq->PrintData();
    //CGNode *rqq = new CGNode(qq, 1);
    //CGNode *rwq = new CGNode(wq, 1);
    //CGNode *req = new CGNode(std::vector<CGNode*>{rqq,rwq},"Add", 1);
    //req->Forward();
//
    //rqq->NodeContent->PrintData();
    //rwq->NodeContent->PrintData();
    //req->NodeContent->PrintData();
    //req->Backward("Output",lq);
    //req->DerivativeNode->NodeContent->PrintData();
    ////rf->Gradient->PrintData();
}
