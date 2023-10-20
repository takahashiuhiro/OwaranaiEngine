#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include "Code/AutomaticDifferentiation/AutoDiffCommon.h"
#include <cmath>
#include <fstream>
#include "Code/AutomaticDifferentiation/Layers/BaseLayer.h"
#include "Code/AutomaticDifferentiation/Layers/LinearSoftmaxLayer.h"
#include "Code/CommonDataStructure/CommonFuncHelpers.h"
#include "Code/AutomaticDifferentiation/Optimizer/SGDOptimizer.h"
#include "Code/AutomaticDifferentiation/Loss/MSELoss.h"
int main() 
{

    std::vector<float> DataSetX = {1,2,3,4,5,6,7,8};//数据集
    std::vector<float> DataSetY = {7,14,20,28,36,42,50,56};

    LinearSoftmaxLayer *m = new LinearSoftmaxLayer(nullptr, "gachi",1,{1,1}, 0);//声明网络

    m->RegisterInputNode("x", {1,1});//注册输入节点
    m->RegisterConstNode("y",{1,1});//注册标签节点
    auto ForwardRes = m->Forward({"x"});//建立forward部分的计算图，获取输出节点

    MSELoss* mse = new MSELoss();//注册mseloss.
    mse->CommonInit(m->CG);
    mse->Build({ForwardRes[0]}, {"y"});//建立loss部分的计算图

    m->CG->BackwardMultiBuildGraph(1);//建立backward计算图

    BaseOptimizer* qweddd = new SGDOptimizer();//声明优化器
    qweddd->Init(m->CG);
    for(int a=0;a<8;a++)
    {
        m->CG->GetNode("x")->AssignContent(new Tensor({1,1},1));//传入输入数据
        m->CG->GetNode("x")->GetContent()->FillArray(DataSetX[a]);
        m->CG->GetNode("y")->AssignContent(new Tensor({1,1},1));//传入输出数据
        m->CG->GetNode("y")->GetContent()->FillArray(DataSetY[a]);
        mse->Backward();//误差反向传播
        m->CG->GetNode(ForwardRes[0])->PrintData();//打印前向结果
        qweddd->Update();//优化器更新
        m->CG->ClearWeightConstExclude();//清理计算图中权重和常量外节点的张量
    }
}
