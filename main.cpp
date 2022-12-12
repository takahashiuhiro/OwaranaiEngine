#include "OwaranaiEngine/OwaranaiEngineInclude.h"

int main()
{
    std::cout<<"------------------------GPU test---------------------------------"<<std::endl;

    /**创建一个线性模块*/
    LinearLayer2D* LinearBlock = new LinearLayer2D();
    /**参数输入*/
    LinearBlock->Params.Set("InputShape", 3, std::vector<size_t>{5,6});
    LinearBlock->Params.Set("OutputShape", 3, std::vector<size_t>{1,8});
    LinearBlock->Params.Set("BatchSize", 3, std::vector<size_t>{2});
    /**参数读取*/
    size_t BatchSize = (*(LinearBlock->Params).Get<std::vector<size_t>>("BatchSize"))[0];
    std::vector<size_t>*InputShape = (LinearBlock->Params).Get<std::vector<size_t>>("InputShape");
    std::vector<size_t>*OutputShape = (LinearBlock->Params).Get<std::vector<size_t>>("OutputShape");
    /**读取输入矩阵*/
    Tensor* q =new Tensor(std::vector<size_t>{BatchSize,(*InputShape)[0],(*InputShape)[1]}, "GPU", 0);
    q->FillArray(3.);
    CGNode *tuq = new CGNode(q, 1);
    /**初始化参数矩阵*/
    Tensor* w =new Tensor(std::vector<size_t>{BatchSize,(*OutputShape)[0],(*InputShape)[0]}, "GPU", 0);
    w->FillArray(5.);
    CGNode *tuw = new CGNode(w, 1);
    Tensor* e =new Tensor(std::vector<size_t>{BatchSize,(*InputShape)[1],(*OutputShape)[1]}, "GPU", 0);
    e->FillArray(7.);
    CGNode *tue = new CGNode(e, 1);
    /**为网络设置输入矩阵和参数矩阵*/
    LinearBlock->InputCGNode = std::vector<CGNode*>{tuq};
    LinearBlock->ParamsCGNode = std::vector<CGNode*>{tuw, tue};
    //设置loss
    LinearBlock->SetLossFunction("MSE");
    Tensor* LabelTensor =new Tensor(std::vector<size_t>{BatchSize,(*OutputShape)[0],(*OutputShape)[1]}, "GPU", 0);
    LabelTensor->FillArray(3149.);
    CGNode* LabelNode = new CGNode(LabelTensor, 0);
    /**先得到前向的结果才能构建loss，因为loss运算需要前向的结果*/
    /**前向图构建*/
    LinearBlock->ForwardBuild();
    /**执行前向运算*/
    LinearBlock->Forward();
    LinearBlock->ForwardNode->NodeContent->PrintData();
    /**输入loss所需的参数*/
    LinearBlock->LossFunctionPointer->OutputNode.push_back(LinearBlock->ForwardNode);
    LinearBlock->LossFunctionPointer->LabelNode.push_back(LabelNode);
    LinearBlock->LossFunctionPointer->LossBuild();
    /**执行前向(训练版)*/
    LinearBlock->LossFunctionPointer->LossNode->Forward();
    //LinearBlock->LossFunctionPointer->LossNode->NodeContent->PrintData();
    /**执行反向*/
    LinearBlock->Backward(LinearBlock->LossFunctionPointer->LossNode->NodeContent);
    LinearBlock->LossFunctionPointer->LossNode->NodeContent->PrintData();

    for(int a=0;a<LinearBlock->ParamsCGNode.size();a++)
    {
        LinearBlock->ParamsCGNode[a]->DerivativeNode->NodeContent->PrintData();
    }
}
