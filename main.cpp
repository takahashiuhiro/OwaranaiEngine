#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 
    Segment e = Segment(Vector(0,0), Vector(10,0));
    Segment q = Segment(Vector(10,1), Vector(0,-9));
    Vector Res;
    if(SegmentCross(e,q,Res))
    {
        Res.PrintData();
    }
}
