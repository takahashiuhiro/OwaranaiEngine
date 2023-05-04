#include "Plane.h"

Plane::Plane(Vector AnchorPosition, Vector NormalVector)
{
    this->AnchorPosition = AnchorPosition;
    this->NormalVector = NormalVector.DirectionVector();
    double MapLen = NormalVector*Vector(0,0,1);
    Vector RandomPositionOnPlane;
    if(BinaryExp<double>(MapLen - NormalVector.Length(), 2) < 1e-8)
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

Plane::Plane(Vector AnchorPosition, Vector BaseVectorFirst, Vector BaseVectorSecond)
{
    this->AnchorPosition = AnchorPosition;
    BaseVector[0] = BaseVectorFirst.DirectionVector();
    BaseVector[1] = BaseVectorSecond.DirectionVector();
    this->NormalVector = BaseVector[0].CrossProduct(BaseVector[1]).DirectionVector();
}

Vector Plane::VectorProjectionToPlane(Vector PositionA)
{
    double FromAnchorToPositionA = (PositionA - AnchorPosition)*NormalVector;
    return PositionA - NormalVector*FromAnchorToPositionA;
}

bool Plane::PlaneCross(Plane& First, Line& ResultLine)
{
    double NormalVectorDotResult = abs(NormalVector*First.NormalVector);
    if(BinaryExp<double>(NormalVectorDotResult - First.NormalVector.Length()*NormalVector.Length(), 2) < 1e-8)
    {
        return 0;
    }
    Vector LineNormalVector = NormalVector.CrossProduct(First.NormalVector).DirectionVector();
    
    int ZeroDimCount = 0;
    for(int a=0;a<LineNormalVector.ShapeCount;a++)
    {
        ZeroDimCount += (LineNormalVector.X() != 0);
    }
    if(!ZeroDimCount)
    {
        return 0;
    }
    if(ZeroDimCount == 1)
    {

    }
}