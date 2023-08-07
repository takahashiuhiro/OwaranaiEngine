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
    w->RegisterVariableNode("e");
    /**声明算子.*/
    w->RegisterOps("c", std::vector<std::string>{"a", "b"}, OpsType::EleMul, Dict());
    w->GetCGOps("c")->SetAddWeight({{"a",3}, {"b", 2}});
    w->RegisterOps("e", std::vector<std::string>{"c", "a"}, OpsType::EleMul, Dict());
    w->GetCGOps("e")->SetAddWeight({{"c",7}, {"a", 5}});

    /**变量赋值.*/
    w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("a")->GetContent()->FillArray(1.5);

    w->GetNode("b")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("b")->GetContent()->FillArray(1.2);

    w->PrintGraphAdjacencyList(2);
    w->BackwardMultiBuildGraph(1);
    w->PrintGraphAdjacencyList(2);

    w->GetNode("e_d")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("e_d")->GetContent()->FillArray(1);

    w->ForwardDfs("a_d");
    w->GetNode("a_d")->PrintData();
    w->ForwardDfs("b_d");
    w->GetNode("b_d")->PrintData();

    return 0;
}