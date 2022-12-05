#include "LinearLayer2D.h"

void LinearLayer2D::ForwardBuild()
{
    CGNode *Intermediate_1 = new CGNode(std::vector<CGNode*>{ParamsCGNode[0],InputCGNode[0]},"Matmul", 1);
    CGNode *Intermediate_2 = new CGNode(std::vector<CGNode*>{Intermediate_1,ParamsCGNode[0]},"Matmul", 1);
    ForwardNode = Intermediate_2;
}