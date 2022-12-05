#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;
    Tensor* loss =new Tensor(std::vector<size_t>{2,1}, "GPU", 0);
    loss->SetV(std::vector<size_t>{0,0}, 100.);
    loss->SetV(std::vector<size_t>{1,0}, 10000.);
    Tensor* q =new Tensor(std::vector<size_t>{2,2,3}, "GPU", 0);
    q->FillArray(3.);
    Tensor* w =new Tensor(std::vector<size_t>{2,3,2}, "GPU", 0);
    w->FillArray(5.);
    Tensor* e =new Tensor(std::vector<size_t>{2,2,3}, "GPU", 0);
    e->FillArray(7.);
    CGNode *tuq = new CGNode(q, 1);
    CGNode *tuw = new CGNode(w, 1);
    CGNode *tue = new CGNode(e, 1);
    CGNode *tuf = new CGNode(std::vector<CGNode*>{tuq,tuw},"Matmul", 1);
    CGNode *tug = new CGNode(std::vector<CGNode*>{tuf,tue},"Matmul", 1);
    CGNode *tuh = new CGNode(std::vector<CGNode*>{tuq, tug},"Elemul", 1);

    tuh->Forward();
    tuh->Backward(loss);
    tuq->DerivativeNode->Forward();
    tuw->DerivativeNode->Forward();
    tue->DerivativeNode->Forward();
    //tuq->DerivativeNode->NodeContent->PrintData();
    //tuf->DerivativeNode->NodeContent->PrintData();
    //tuf->NodeContent->PrintData();
    //tug->DerivativeNode->NodeContent->PrintData();
    //tug->NodeContent->PrintData();
    tuh->ClearGradient(std::vector<CGNode*>{tuq,tuw,tue});

    Hyperparameter ttets;
    ttets.Set("qwe", 2, std::vector<std::string>{"qdd", "err"});
    std::vector<std::string>*pp = ttets.Get<std::vector<std::string>>("qwe");
    for(int a =0 ;a<pp->size();a++)
    {
        std::cout<<(*pp)[a]<<std::endl;
    }
    //std::cout<<HyperparameterTypeConst::FLOAT<<std::endl;
}
