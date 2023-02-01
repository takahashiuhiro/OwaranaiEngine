#include "GeometryHelpers.h"

/**线段相交.
 * 只计算x,y不计算z，如果需要在非水平面计算需要先转换坐标系, 不计算多个交点
 * Params: 线段1, 线段2, 交点
*/
bool SegmentCross(Segment& First, Segment& Second, Vector& ResultPosition)
{
    /**头尾相接.*/
    if((First.StartPoint - Second.StartPoint).Length() < 1e-8 || (First.StartPoint - Second.EndPoint).Length() < 1e-8 ||(First.EndPoint - Second.StartPoint).Length() < 1e-8 ||(First.EndPoint - Second.EndPoint).Length() < 1e-8)
    {
        if((First.StartPoint - Second.StartPoint).Length() < 1e-8)ResultPosition = Second.StartPoint;
        if((First.StartPoint - Second.EndPoint).Length() < 1e-8)ResultPosition = First.StartPoint;
        if((First.EndPoint - Second.StartPoint).Length() < 1e-8)ResultPosition = Second.StartPoint;
        if((First.EndPoint - Second.EndPoint).Length() < 1e-8)ResultPosition = Second.EndPoint;
        return 1;
    }
    /**平行.*/
    if((First.DirectionVector() - Second.DirectionVector()).Length() < 1e-8 || (First.DirectionVector() + Second.DirectionVector()).Length() < 1e-8)
    {
        return 0;
    }
    /**轮换对称的检测一条线段的起点和终点是否在另一条线段的同一侧，如果是就不相交.*/
    Vector ProtoSegment,StToSt, StToEd;
    for(int a=0;a<2;a++)
    {
        if(a)
        {
            ProtoSegment = First.DirectionVector();
            StToSt = Second.StartPoint - First.StartPoint;
            StToEd = Second.EndPoint - First.StartPoint;
        }
        else
        {
            ProtoSegment = Second.DirectionVector();
            StToSt = First.StartPoint - Second.StartPoint;
            StToEd = First.EndPoint - Second.StartPoint;
        }
        if(ProtoSegment.CrossProduct(StToSt).Z * ProtoSegment.CrossProduct(StToEd).Z > 0)
        {
            return 0;
        }
    }
    Vector StToStProjection = ProtoSegment*(ProtoSegment * StToSt);
    Vector StToEdProjection = ProtoSegment*(ProtoSegment * StToEd);
    Vector BiasPosition = First.StartPoint + StToStProjection;
    Vector ProjectionVec = StToEdProjection - StToStProjection;
    double ProjectionVecPercent = (StToStProjection-StToSt).Length()/((StToStProjection-StToSt).Length() + (StToEd - StToEdProjection).Length());
    //下面都不对，别信，睡了，有空改
    ResultPosition = BiasPosition + ProjectionVec*ProjectionVecPercent;
    return 1;
}