#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
int main() {
    /**建图.*/
    ComputationalGraph* w = new ComputationalGraph();

    /**声明变量*/
    w->RegisterWeightNode("a");//声明权重变量
    w->RegisterVariableNode("b");
    w->RegisterVariableNode("c");
    w->RegisterVariableNode("d");
    w->RegisterVariableNode("e");
    w->RegisterVariableNode("f");
    w->RegisterVariableNode("g");

    /**声明算子.*/
    w->RegisterOpsCompleted("c", std::vector<std::string>{"a", "b"}, OpsType::MatMul, Dict());//c = a@b
    w->RegisterOpsCompleted("d", std::vector<std::string>{"c", "a"}, OpsType::MatMul, Dict());
    w->RegisterOpsCompleted("e", std::vector<std::string>{"d", "b"}, OpsType::MatMul, Dict());
    w->RegisterOpsCompleted("f", std::vector<std::string>{"e", "g"}, OpsType::MatMul, Dict());

    /**求导.*/
    w->BackwardMultiBuildGraph(1);

    for(int a=0;a<50;a++)
    {
        float x = 1;
        if(a%2)x+=3;//用于输出不一样的张量

        /**清理权重，常量以外的变量.*/
        w->ClearDataPropertyExclude();

        w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{1,3},0));//对节点a赋值一个张量
        w->GetNode("a")->GetContent()->FillArray(1.5*x);//对节点a的张量，填充数据

        w->GetNode("b")->AssignContent(new Tensor(std::vector<size_t>{3,1},0));
        w->GetNode("b")->GetContent()->FillArray(1.2*x);

        w->GetNode("g")->AssignContent(new Tensor(std::vector<size_t>{1,1},0));
        w->GetNode("g")->GetContent()->FillArray(1.9*x);

        w->GetNode("f_d")->AssignContent(new Tensor(std::vector<size_t>{1,1},0));//对节点f的导数赋值张量
        w->GetNode("f_d")->GetContent()->FillArray(1.*x);//对节点f的张量，填充数据

        w->ForwardDfs("a_d");//对节点a求导
        w->GetNode("a_d")->PrintData();//打印节点a的导数
    }

    return 0;
}