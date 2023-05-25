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
    //通过法向量叉乘得到交线的方向
    Vector LineNormalVector = NormalVector.CrossProduct(First.NormalVector).DirectionVector();
    ResultLine.DirctionVector = LineNormalVector;
    //把First平面的锚点映射到this平面上
    Vector FirstAnchorMapToThis = VectorProjectionToPlane(First.AnchorPosition);
    Vector ConnectVector = FirstAnchorMapToThis - First.AnchorPosition;
    //如果映射的点和锚点零距离，那说明这就是交线的锚点。
    if(ConnectVector.Length() < 1e-8)
    {
        ResultLine.AnchorPosition = FirstAnchorMapToThis;
        return 1;
    }
    //用连接的向量叉乘交线的方向向量得到映射点指向交线的方向...也可能是反方向
    Vector MapPointToLine = ConnectVector.CrossProduct(LineNormalVector).DirectionVector();
    //计算映射点到交线的距离
    double MapPointToLineLen = abs(ConnectVector.Length()/std::tan(std::acos(NormalVector*First.NormalVector/(NormalVector.Length()*First.NormalVector.Length()))));
    Vector TryPoint = FirstAnchorMapToThis - MapPointToLine*MapPointToLineLen;
    if(First.VectorProjectionToPlane(TryPoint).Length() < 1e-8)
    {
        ResultLine.AnchorPosition = TryPoint;
        return 1;
    }
    ResultLine.AnchorPosition = FirstAnchorMapToThis + MapPointToLine*MapPointToLineLen;
    return 1;
}