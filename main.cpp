#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 
    Segment t = Segment(Vector(0,0), Vector(0,2));
    Segment y = Segment(Vector(-1,1), Vector(1,1));
    Vector res;
    std::cout<<t.SegmentCross2D(y,res)<<std::endl;
    res.PrintData();
    std::cout<<BinaryExp<long long >(2.999,10)<<std::endl;
}

