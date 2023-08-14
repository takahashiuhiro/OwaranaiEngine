#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
int main() {
    /**建图.*/
    ComputationalGraph* w = new ComputationalGraph();

    /**声明变量*/
    w->RegisterVariableNode("a");
    w->RegisterVariableNode("b");
    w->RegisterVariableNode("c");
    w->RegisterVariableNode("d");
    w->RegisterVariableNode("e");
    w->RegisterVariableNode("f");
    w->RegisterVariableNode("g");

    /**声明算子.*/
    w->RegisterOpsCompleted("c", std::vector<std::string>{"a", "b"}, OpsType::MatMul, Dict());
    w->RegisterOpsCompleted("d", std::vector<std::string>{"c", "a"}, OpsType::MatMul, Dict());
    w->RegisterOpsCompleted("e", std::vector<std::string>{"d", "b"}, OpsType::MatMul, Dict());
    w->RegisterOpsCompleted("f", std::vector<std::string>{"e", "g"}, OpsType::MatMul, Dict());

    w->BackwardMultiBuildGraph(1);

    for(int a=0;a<50;a++)
    {

        w->ClearAllData();

        float x = 1;
        if(a%2)x+=3;

        /**变量赋值.*/
        w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{1,3},0));
        w->GetNode("a")->GetContent()->FillArray(1.5*x);

        w->GetNode("b")->AssignContent(new Tensor(std::vector<size_t>{3,1},0));
        w->GetNode("b")->GetContent()->FillArray(1.2*x);

        w->GetNode("g")->AssignContent(new Tensor(std::vector<size_t>{1,1},0));
        w->GetNode("g")->GetContent()->FillArray(1.9*x);

        w->GetNode("f_d")->AssignContent(new Tensor(std::vector<size_t>{1,1},0));
        w->GetNode("f_d")->GetContent()->FillArray(1.*x);

        w->ForwardDfs("a_d");
        w->GetNode("a_d")->PrintData();

    }

    return 0;
}