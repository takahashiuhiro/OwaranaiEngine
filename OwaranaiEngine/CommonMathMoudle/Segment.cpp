#include "Segment.h"

Segment::Segment(Vector StartPoint, Vector EndPoint)
{
    this->StartPoint = StartPoint;
    this->EndPoint = EndPoint;
}

Vector Segment::DirectionVector()
{
    Vector SegmentVec = EndPoint - StartPoint;
	double VectorLength = SegmentVec.Length();
	if (VectorLength * VectorLength < 1e-8) return Vector(0, 0);
	return SegmentVec / VectorLength;
}

Vector Segment::GetPosition(double LengthPercent)
{
    Vector SegmentVec = EndPoint - StartPoint;
    return StartPoint + SegmentVec*LengthPercent;
}

bool Segment::SegmentCross2D(Segment& Input, Vector& ResultPosition)
{
    Segment& First = (*this);
    /**头尾相接.*/
    if((First.StartPoint - Input.StartPoint).Length() < 1e-8 || (First.StartPoint - Input.EndPoint).Length() < 1e-8 ||(First.EndPoint - Input.StartPoint).Length() < 1e-8 ||(First.EndPoint - Input.EndPoint).Length() < 1e-8)
    {
        if((First.StartPoint - Input.StartPoint).Length() < 1e-8)ResultPosition = Input.StartPoint;
        if((First.StartPoint - Input.EndPoint).Length() < 1e-8)ResultPosition = First.StartPoint;
        if((First.EndPoint - Input.StartPoint).Length() < 1e-8)ResultPosition = Input.StartPoint;
        if((First.EndPoint - Input.EndPoint).Length() < 1e-8)ResultPosition = Input.EndPoint;
        return 1;
    }
    /**平行.*/
    if((First.DirectionVector() - Input.DirectionVector()).Length() < 1e-8 || (First.DirectionVector() + Input.DirectionVector()).Length() < 1e-8)
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
            StToSt = Input.StartPoint - First.StartPoint;
            StToEd = Input.EndPoint - First.StartPoint;
        }
        else
        {
            ProtoSegment = Input.DirectionVector();
            StToSt = First.StartPoint - Input.StartPoint;
            StToEd = First.EndPoint - Input.StartPoint;
        }
        if(ProtoSegment.CrossProduct(StToSt).Z() * ProtoSegment.CrossProduct(StToEd).Z() > 0)
        {
            return 0;
        }
    }
    Vector StToStProjection = ProtoSegment*(ProtoSegment * StToSt);
    Vector StToEdProjection = ProtoSegment*(ProtoSegment * StToEd);
    Vector BiasPosition = First.StartPoint + StToStProjection;
    Vector ProjectionVec = StToEdProjection - StToStProjection;
    double ProjectionVecPercent = (StToStProjection-StToSt).Length()/((StToStProjection-StToSt).Length() + (StToEd - StToEdProjection).Length());
    ResultPosition = BiasPosition + ProjectionVec*ProjectionVecPercent;
    return 1;
}