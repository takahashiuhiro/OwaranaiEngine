#pragma once
#include "StdInclude.h"
#include "MoudleInclude.h"

struct Segment
{
public:
    Segment(){};
    Segment(Vector StartPoint, Vector EndPoint)
    {
        this->StartPoint = StartPoint;
        this->EndPoint = EndPoint;
    }

    Vector StartPoint = Vector(0,0,0);
    Vector EndPoint = Vector(0,0,0);

    /**方向向量.*/
	Vector DirectionVector()
	{
        Vector SegmentVec = EndPoint - StartPoint;
		double VectorLength = SegmentVec.Length();
		if (VectorLength * VectorLength < 1e-8) return Vector(0, 0);
		return SegmentVec / VectorLength;
	}
};

