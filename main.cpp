#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 
    Tensor* qee = new Tensor(std::vector<size_t>{2,4,3});
    Tensor* qwe = new Tensor(std::vector<size_t>{2,2,3});
    qwe->FillArray(3);
    qee->FillArray(2);
    Tensor* eee = qee->TensorSplice(qwe, 1);
    eee->PrintData();

    Tensor* wqee = new Tensor(std::vector<size_t>{2,4,3});
    Tensor* wqwe = new Tensor(std::vector<size_t>{2,2,3});
    wqwe->FillArray(3);
    wqee->FillArray(2);
    Tensor* weee = wqee->TensorSplice(wqwe, 1);
    weee->PrintData();

    weee->GetUnitTensor(std::vector<size_t>{2,8,8})->PrintData();
}
