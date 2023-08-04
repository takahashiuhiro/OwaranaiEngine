//#include"Code/CommonDataStructure/BaseGraph.h"
//#include"Code/CommonDataStructure/Dict.h"
#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"

int main() {

    ComputationalGraph* w = new ComputationalGraph();

    w->RegisterNode("a");
    w->GetNode("a")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("a")->GetContent()->FillArray(1.5);
    w->GetNode("a")->Property.Set("RequireGrad", true);
    w->GetNode("a")->Property.Set("Input", true);
    w->GetNode("a")->GetContent()->PrintData();

    w->RegisterNode("b");
    w->GetNode("b")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("b")->GetContent()->FillArray(1.2);
    w->GetNode("b")->Property.Set("RequireGrad", true);
    w->GetNode("b")->Property.Set("Input", true);
    w->GetNode("b")->GetContent()->PrintData();

    w->RegisterNode("c");
    w->GetNode("c")->Property.Set("RequireGrad", true);
    w->GetNode("c")->Property.Set("Input", true);


    w->RegisterOps("c", std::vector<std::string>{"a", "b"}, OpsType::Add, Dict());
    auto &OutDNodeOpsParamsAddWeight = *(w->Opss["c"]->Params.template Get<std::shared_ptr<std::map<std::string, float>>>(std::string("AddWeight")));
    OutDNodeOpsParamsAddWeight["a"] = 4;
    OutDNodeOpsParamsAddWeight["b"] = 2;

    w->RegisterNode("d");
    w->GetNode("d")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("d")->GetContent()->FillArray(10);
    w->GetNode("d")->Property.Set("RequireGrad", true);
    w->GetNode("d")->Property.Set("Input", true);
    w->GetNode("d")->GetContent()->PrintData();

    w->RegisterNode("e");
    w->GetNode("e")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("e")->GetContent()->FillArray(1000);
    w->GetNode("e")->Property.Set("RequireGrad", true);
    w->GetNode("e")->Property.Set("Input", true);
    w->GetNode("e")->GetContent()->PrintData();

    w->RegisterOps("e", std::vector<std::string>{"c", "d"}, OpsType::Add, Dict());
    auto &OutDNodeOpsParamsAddWeight11 = *(w->Opss["e"]->Params.template Get<std::shared_ptr<std::map<std::string, float>>>(std::string("AddWeight")));
    OutDNodeOpsParamsAddWeight11["c"] = 2.8;
    OutDNodeOpsParamsAddWeight11["d"] = 11111;

    w->BackwardGraphBuild();


    w->GetNode("e_d")->AssignContent(new Tensor(std::vector<size_t>{2,3},1));
    w->GetNode("e_d")->GetContent()->FillArray(10);


    w->ForwardDfs("d_d");
    w->GetNode("d_d")->GetContent()->PrintData();


    return 0;
}