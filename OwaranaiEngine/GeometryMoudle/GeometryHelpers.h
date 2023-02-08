#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"

/**2D线段相交.
 * 只计算x,y不计算z，如果需要在非水平面计算需要先转换坐标系, 不计算多个交点
 * Params: 线段1, 线段2, 交点
*/
bool SegmentCross2D(Segment& First, Segment& Second, Vector& ResultPosition);
/**
 * 平面相交
*/
//bool PlaneCross(Plane& First, Plane& Second, Line& ResultLine);