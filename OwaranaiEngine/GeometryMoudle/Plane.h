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
        double MapLen = NormalVector*Vector(0,0,1);
        Vector RandomPositionOnPlane;
        if((MapLen - NormalVector.Length())*(MapLen - NormalVector.Length()) < 1e-8)
        {
            RandomPositionOnPlane = AnchorPosition + Vector(0,1,0);
        }
        else
        {
            RandomPositionOnPlane = AnchorPosition + Vector(0,0,1);
        }
        Vector VectorOnPlane = VectorProjectionToPlane(RandomPositionOnPlane).DirectionVector();
        Vector AnotherVectorOnPlane = VectorOnPlane.CrossProduct(NormalVector).DirectionVector();
        if(VectorOnPlane.CrossProduct(AnotherVectorOnPlane)*NormalVector > 0)
        {
            BaseVector[0] = VectorOnPlane;
            BaseVector[1] = AnotherVectorOnPlane;
        }
        else
        {
            BaseVector[0] = AnotherVectorOnPlane;
            BaseVector[1] = VectorOnPlane;
        }
    }

    /**
     * 通过锚点和正交基得到平面
     * Params:锚点，单位向量1，单位向量2
    */
    Plane(Vector AnchorPosition, Vector BaseVectorFirst, Vector BaseVectorSecond)
    {
        this->AnchorPosition = AnchorPosition;
        BaseVector[0] = BaseVectorFirst.DirectionVector();
        BaseVector[1] = BaseVectorSecond.DirectionVector();
        this->NormalVector = BaseVector[0].CrossProduct(BaseVector[1]);
    }

    /**锚定点.*/
    Vector AnchorPosition = Vector(0,0,0);
    /**法向量方向.*/
    Vector NormalVector = Vector(0,1,0);
    /**平面上当前的正交基.*/
    Vector BaseVector[2] = {Vector(0,1,0), Vector(1,0,0)};

    /**
     * 空间中一点在平面上的投影
     * Params:空间中的一个点
    */
    Vector VectorProjectionToPlane(Vector PositionA)
    {
        double FromAnchorToPositionA = (PositionA - AnchorPosition)*NormalVector;
        return PositionA - NormalVector*FromAnchorToPositionA;
    }
};
