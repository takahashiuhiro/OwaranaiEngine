#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"

struct Plane
{
public:
    Plane(){}
    /**通过锚定点和法线方向定义平面，求出一组可用的正交基.*/
    Plane(Vector AnchorPosition, Vector NormalVector);

    /**
     * 通过锚点和正交基得到平面
     * Params:锚点，单位向量1，单位向量2
    */
    Plane(Vector AnchorPosition, Vector BaseVectorFirst, Vector BaseVectorSecond);

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
    Vector VectorProjectionToPlane(Vector PositionA);

    /**平面相交
     * Params: 平面,交线
    */
    bool PlaneCross(Plane& First, Line& ResultLine);
    
};
