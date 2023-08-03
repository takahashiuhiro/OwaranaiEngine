//#include"Code/CommonDataStructure/BaseGraph.h"
//#include"Code/CommonDataStructure/Dict.h"
#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"

int main() {

    ComputationalGraph* w = new ComputationalGraph();

    w->RegisterNode("a");
    w->GetNode("a")->Content = new Tensor(std::vector<size_t>{2,3},1);
    w->GetNode("a")->Content->FillArray(1.5);
    w->GetNode("a")->Content->PrintData();

    w->RegisterNode("b");
    w->GetNode("b")->Content = new Tensor(std::vector<size_t>{2,3},1);
    w->GetNode("b")->Content->FillArray(1.2);
    w->GetNode("b")->Content->PrintData();

    w->RegisterNode("c");


    w->RegisterOps("c", std::vector<std::string>{"a", "b"}, OpsType::Add, Dict());
    auto &OutDNodeOpsParamsAddWeight = *(w->Opss["c"]->Params.template Get<std::shared_ptr<std::map<std::string, float>>>(std::string("AddWeight")));
    OutDNodeOpsParamsAddWeight["a"] = 4;
    OutDNodeOpsParamsAddWeight["b"] = 2;

    w->Opss["c"]->Forward();
    w->GetNode("c")->Content->PrintData();

    w->GetNode("a")->Property.Set("RequireGrad", true);
    w->GetNode("b")->Property.Set("RequireGrad", true);
    w->GetNode("c")->Property.Set("RequireGrad", true);
    w->GetNode("a")->Property.Set("Input", true);
    w->GetNode("b")->Property.Set("Input", true);
    w->GetNode("c")->Property.Set("Input", true);


    w->BackwardGraphBuild();

    w->GetNode("c_d")->Content = new Tensor(std::vector<size_t>{2,3},1);
    w->GetNode("c_d")->Content->FillArray(10);


    w->Opss["b_d"]->Forward();
    w->GetNode("b_d")->GetContent()->PrintData();
    w->Opss["a_d"]->Forward();
    w->GetNode("a_d")->GetContent()->PrintData();

    return 0;
}