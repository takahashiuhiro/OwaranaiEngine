#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
int main() {
    /**建图.*/
    ComputationalGraph* w = new ComputationalGraph();
    /**声明变量a.*/
    w->RegisterVariableNode("a");
    w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("a")->GetContent()->FillArray(1.5);
    //->GetNode("a")->PrintData();
    /**声明变量b.*/
    w->RegisterVariableNode("b");
    w->GetNode("b")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("b")->GetContent()->FillArray(1.2);
    //w->GetNode("b")->PrintData();
    /**声明变量c.*/
    w->RegisterVariableNode("c");
    /**c=4*a+2*b.*/
    w->RegisterOps("c", std::vector<std::string>{"a", "b"}, OpsType::Add, Dict());
    w->GetCGOps("c")->SetAddWeight({{"a",4}, {"b", 2}});
    /**声明变量d.*/
    w->RegisterVariableNode("d");
    w->GetNode("d")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("d")->GetContent()->FillArray(10);
    //w->GetNode("d")->PrintData();
    /**声明变量e.*/
    w->RegisterVariableNode("e");
    w->GetNode("e")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("e")->GetContent()->FillArray(1000);
    //w->GetNode("e")->PrintData();
    /**e=2.8*c+11111*d.*/
    w->RegisterOps("e", std::vector<std::string>{"c", "d"}, OpsType::Add, Dict());
    w->GetCGOps("e")->SetAddWeight({{"c",2.8}, {"d", 11111}});
    /**建立反向图.*/
    w->PrintGraphAdjacencyList(2);
    w->BackwardMultiBuildGraph(3);
    w->PrintGraphAdjacencyList(2);
    /**给输出节点的导数赋值.*/
    w->GetNode("e_d")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("e_d")->GetContent()->FillArray(10);
    /**求a的导数.*/
    w->ForwardDfs("a_d");
    //w->GetNode("a_d")->PrintData();


    return 0;
}