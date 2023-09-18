#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include "Code/AutomaticDifferentiation/AutoDiffCommon.h"
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

    ComputationalNodeInfo qa = ComputationalNodeInfo("{Ops:1,Name:{Ops:1,Name:{Ops:1,Name:{Ops:1,Name:asd,D:0,ExtraInfo:tutututu,},D:0,ExtraInfo:tutututu,},D:0,ExtraInfo:tutututu,},D:0,ExtraInfo:{Ops:1,Name:{Ops:1,Name:{Ops:1,Name:{Ops:1,Name:asd,D:0,ExtraInfo:tutututu,},D:0,ExtraInfo:tutututu,},D:0,ExtraInfo:tutututu,},D:0,ExtraInfo:tutututu,},}");

    std::cout<<qa.D<<std::endl;
    std::cout<<qa.Ops<<std::endl;
    std::cout<<qa.ExtraInfo<<std::endl;
    std::cout<<qa.Name<<std::endl;

    std::string tt = qa.DumpToString();
    std::cout<<qa.DumpToString()<<std::endl;
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(tt);
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(qa.DumpToString());
    qa = ComputationalNodeInfo(qa.DumpToString());
    std::cout<<qa.DumpToString()<<std::endl;
    std::cout<<(qa.DumpToString() == tt)<<std::endl;
}


