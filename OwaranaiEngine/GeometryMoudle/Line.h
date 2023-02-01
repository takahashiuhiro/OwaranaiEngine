#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"

struct Line
{
public:
    Line(){}
    Line(Vector AnchorPosition, Vector DirctionVector)
    {
        this->AnchorPosition = AnchorPosition;
        this->DirctionVector = DirctionVector.DirectionVector();
    }

    /**锚定点.*/
    Vector AnchorPosition = Vector(0,0,0);
    /**直线方向.*/
    Vector DirctionVector = Vector(0,0,0);
};
