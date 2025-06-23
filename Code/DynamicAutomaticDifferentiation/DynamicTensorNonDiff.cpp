#include "DynamicTensor.h"

DynamicTensor DynamicTensor::SampleFromMulGaussian(DynamicTensor MeanVector, DynamicTensor CovarianceMatrix, std::vector<int>OutputShape, bool RequiresGrad, size_t DeviceNum, int Seed)
{
    if (Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::vector<size_t>ConvOutputShape;
    for(auto&it:OutputShape)ConvOutputShape.push_back(it);
    Tensor* TensorData = Tensor::GenMulGaussianDistribution(MeanVector.Ops->TensorPointer.get(), CovarianceMatrix.Ops->TensorPointer.get(), ConvOutputShape, Seed, DeviceNum);
    DynamicTensor Res = DynamicTensor(std::shared_ptr<Tensor>(TensorData), RequiresGrad);
    return Res;
}

DynamicTensor DynamicTensor::Abs()
{
    DynamicTensor MinusSelf = (DynamicTensor(Ops)*(-1)).ReLU();
    DynamicTensor Self = DynamicTensor(Ops).ReLU();
    return MinusSelf + Self;
}

DynamicTensor DynamicTensor::GetProbabilityDensityFromGaussian(DynamicTensor MeanVector, DynamicTensor CovarianceMatrix, bool IsDiagonal)
{
    if(!IsDiagonal)
    {
        //还没写非对角版的
        Log::Assert(false, "GetProbabilityDensityFromGaussian::todo");
    }
    else
    {
        // 这里的self应该是个二维的
        DynamicTensor Self = DynamicTensor(Ops);
        DynamicTensor BiasRes = Self - MeanVector.View({1, MeanVector.ShapeCount()});
        DynamicTensor InvVarRes = CovarianceMatrix.Pow(-1).View({1, CovarianceMatrix.ShapeCount()});
        DynamicTensor ExpRes = ((BiasRes*InvVarRes*BiasRes).Sum({1}, true)*(-0.5)).Eleexp(M_E);
        double FirstNum = std::pow(2*M_PI, -0.5*MeanVector.ShapeCount());
        DynamicTensor DetRes = (CovarianceMatrix.EleLog().Sum({0}, true).Eleexp(M_E).Pow(-0.5)*FirstNum).View({1,1});
        return DetRes*ExpRes;
    }
}