#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 
    Plane a = Plane(Vector(0,1,0), Vector(-0.5,0,1));
    Plane b = Plane(Vector(0,1,0), Vector(0.5,0,1));
    Line res;
    std::cout<<a.PlaneCross(b, res)<<std::endl;
    res.AnchorPosition.PrintData();
    res.DirctionVector.PrintData();
}

