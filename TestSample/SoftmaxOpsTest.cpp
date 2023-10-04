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
    w->RegisterVariableNode("a");//声明权重变量
    w->RegisterVariableNode("b");
    w->GetNode("a")->NodeContentShape = {1,1,3};
    //w->GetNode("b")->NodeContentShape = {1,1,3};

    w->RegisterOpsCompleted("b", std::vector<std::string>{"a"}, OpsType::Softmax, Dict());
    w->GetCGOps("b")->SetSelectDim({{"a",2}});
    w->GetCGOps("b")->AfterSettingShapeComputing();

    w->BackwardMultiBuildGraph(1);


    w->GetNode("b_d")->AssignContent(new Tensor(std::vector<size_t>{1,1,3},0));
    w->GetNode("b_d")->GetContent()->FillArray(1);
    w->GetNode("b_d")->GetContent()->SetV({0,0,1}, 2);
    w->GetNode("b_d")->GetContent()->SetV({0,0,2}, 1);

    w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{1,1,3},0));//对节点f的导数赋值张量
    w->GetNode("a")->GetContent()->FillArray(1);//对节点f的张量，填充数据
    w->GetNode("a")->GetContent()->SetV({0,0,1}, 2);
    w->GetNode("a")->GetContent()->SetV({0,0,2}, 3);

    w->ForwardDfs("a_d");
    w->GetNode("a")->PrintData();
    w->GetNode("b")->PrintData();
    w->GetNode("a_d")->PrintData();


    //std::cout<<q->CanBroadCastTo({3,3,3})<<std::endl;


}

