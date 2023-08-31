#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include <cmath>
int main() 
{
    /**建图.*/
    ComputationalGraph* w = new ComputationalGraph();

    /**声明变量*/
    w->RegisterWeightNode("a");//声明权重变量
    w->RegisterVariableNode("b");

    w->RegisterOpsCompleted("b", std::vector<std::string>{"a"}, OpsType::Softmax, Dict());
    w->GetCGOps("b")->SetSelectDim({{"a",1}});

    w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{2,3},0));//对节点f的导数赋值张量

    w->GetNode("a")->GetContent()->FillArray(1);//对节点f的张量，填充数据
    w->GetNode("a")->GetContent()->SetV({0,1}, 2);

    w->ForwardDfs("b");
    w->GetNode("b")->PrintData();

}
