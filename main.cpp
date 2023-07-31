//#include"Code/CommonDataStructure/BaseGraph.h"
//#include"Code/CommonDataStructure/Dict.h"
#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include"Code/AutomaticDifferentiation/ComputationalGraph.h"

int main() {

    ComputationalGraph* w = new ComputationalGraph();

    w->RegisterNode("a");
    w->GetNode("a")->Content = new Tensor(std::vector<size_t>{2,3},1);
    w->GetNode("a")->Content->FillArray(1.5);

    w->RegisterNode("b");
    w->GetNode("b")->Content = new Tensor(std::vector<size_t>{2,3},1);
    w->GetNode("b")->Content->FillArray(1.2);

    w->RegisterNode("c");


    w->RegisterOps("c", std::vector<std::string>{"a", "b"}, 2, Dict());
    //w->Opss["c"].

    w->Opss["c"]->Forward();

    w->GetNode("c")->Content->PrintData();

    return 0;
}