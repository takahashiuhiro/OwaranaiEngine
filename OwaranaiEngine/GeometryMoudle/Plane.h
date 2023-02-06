#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"

struct Plane
{
public:
    Plane(){}
    /**通过铆钉点和法线方向定义平面，求出一组可用的正交基.*/
    Plane(Vector AnchorPosition, Vector NormalVector)
    {
        this->AnchorPosition = AnchorPosition;
        this->NormalVector = NormalVector;
        Vector RandomPositionOnPlane;
        if(NormalVector.Z*NormalVector.Z < 1e-8)
        {
            
        }
        else
        {

        }
    }

    Plane(Vector BaseVectorFirst, Vector BaseVectorSecond)
    {

    }

    /**锚定点.*/
    Vector AnchorPosition = Vector(0,0,0);
    /**法向量方向.*/
    Vector NormalVector = Vector(0,1,0);
    /**平面上当前的正交基.*/
    Vector BaseVector[2] = {Vector(0,1,0), Vector(1,0,0)};
};
