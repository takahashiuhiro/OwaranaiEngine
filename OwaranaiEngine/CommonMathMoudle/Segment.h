#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"

struct Segment
{
public:
    Segment(){};
    Segment(Vector StartPoint, Vector EndPoint);

    Vector StartPoint = Vector(0,0,0);
    Vector EndPoint = Vector(0,0,0);

    /**方向向量.*/
	Vector DirectionVector();

    /**通过长度百分比得到位置.*/
    Vector GetPosition(double LengthPercent);

    /**2D线段相交.
     * 只计算x,y不计算z，如果需要在非水平面计算需要先转换坐标系, 不计算多个交点
     * Params: 线段1, 线段2, 交点
    */
    bool SegmentCross2D(Segment& Input, Vector& ResultPosition);
};

