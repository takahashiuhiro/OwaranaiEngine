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

    /**声明算子.*/
    w->RegisterOpsCompleted("c", std::vector<std::string>{"a", "b"}, OpsType::MatMul, Dict());
    w->RegisterOpsCompleted("d", std::vector<std::string>{"c", "a"}, OpsType::MatMul, Dict());
    w->RegisterOpsCompleted("e", std::vector<std::string>{"d", "b"}, OpsType::MatMul, Dict());

    /**变量赋值.*/
    w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{1,3},0));
    w->GetNode("a")->GetContent()->FillArray(1.5);

    w->GetNode("b")->AssignContent(new Tensor(std::vector<size_t>{3,1},0));
    w->GetNode("b")->GetContent()->FillArray(1.2);
    

    w->BackwardMultiBuildGraph(1);

    w->GetNode("e_d")->AssignContent(new Tensor(std::vector<size_t>{1,1},0));
    w->GetNode("e_d")->GetContent()->FillArray(1.);

    w->ForwardDfs("a_d");
    w->GetNode("a_d")->PrintData();

    w->ForwardDfs("b_d");
    w->GetNode("b_d")->PrintData();


    return 0;
}