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
    /**执行前向网络构建方法*/
    LinearBlock->ForwardBuild();
    /**执行前向*/
    LinearBlock->Forward();

    //TODO::听我说你先别急，新加了模块现在直接跑肯定挂的，得把↑的前向结果送到loss里以后再搞↓

    /**设置loss*/
    Tensor* loss =new Tensor(std::vector<size_t>{2,1}, "GPU", 0);
    loss->SetV(std::vector<size_t>{0,0}, 100.);
    loss->SetV(std::vector<size_t>{1,0}, 10000.);
    /**执行反向*/
    LinearBlock->Backward(loss);
}
