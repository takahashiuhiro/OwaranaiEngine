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
    LinearSoftmaxLayer *m = new LinearSoftmaxLayer(nullptr, "gachi",1,{1,1}, 0);//声明网络
    m->RegisterInputNode("x", {1,1});//注册输入节点
    auto ForwardRes = m->Forward({"x"});//建立forward部分的计算图，获取输出节点
    MSELoss* mse = new MSELoss();//注册mseloss.
    mse->CommonInit(m->CG);
    m->CG->RegisterConstNode("y",{1,1});//注册标签节点
    mse->Build({ForwardRes[0]}, {"y"});//建立loss部分的计算图
    m->CG->BackwardMultiBuildGraph(1);//建立backward计算图
    BaseOptimizer* qweddd = new SGDOptimizer();//声明优化器
    qweddd->Init(m->CG);
    std::vector<float> DataSetX = {1,2,3,4,5,6,7,8};//数据集
    std::vector<float> DataSetY = {7,14,20,28,36,42,50,56};
    for(int a=0;a<8;a++)
    {
        m->CG->GetNode("gachi.layer_1.LinearWeight")->PrintData();//打印权重矩阵
        m->CG->GetNode("x")->AssignContent(new Tensor({1,1},1));//传入输入数据
        m->CG->GetNode("x")->GetContent()->FillArray(DataSetX[a]);
        m->CG->GetNode("y")->AssignContent(new Tensor({1,1},1));//传入输出数据
        m->CG->GetNode("y")->GetContent()->FillArray(DataSetY[a]);
        m->CG->ForwardDfs(mse->LossNodes[0]);//计算图执行到loss节点
        m->CG->GetNode(m->CG->GetDNodeid(mse->LossNodes[0]))->AssignContent(m->CG->GetNode(mse->LossNodes[0])->GetContent()->Copy());//把loss的结果赋值给他的梯度节点
        m->CG->ComputeWeightNodesDForward();//执行计算图内所有允许求导的节点的反向
        qweddd->SyncTensorByCG();//优化器从计算图中同步权重和他的梯度
        qweddd->Update();//优化器更新
        qweddd->SyncTensorToCG();//优化器把结果传回计算图
        m->CG->ClearWeightConstExclude();//清理计算图中权重和常量外节点的张量
    }
}
