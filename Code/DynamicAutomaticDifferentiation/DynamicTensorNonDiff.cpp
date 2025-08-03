#include "DynamicTensor.h"

DynamicTensor DynamicTensor::Cholesky()
{
    using ContentType = std::decay_t<decltype(*(Ops->TensorPointer))>;
    auto CholeskyRes = std::shared_ptr<ContentType>(Ops->TensorPointer->Cholesky());
    return DynamicTensor(CholeskyRes, Ops->RequiresGrad);
}

DynamicTensor DynamicTensor::SampleFromStdGaussian(int Dim, std::vector<int> InputVec, int Seed,int DeviceNum)
{
    std::vector<size_t> ShapeVec;
    for(auto&it:InputVec)ShapeVec.push_back(it);
    if(Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto ContentRes = Tensor::SampleMultivariateStandardGaussian(Dim, ShapeVec, Seed, DeviceNum);
    using ContentType = std::remove_pointer_t<std::decay_t<decltype(ContentRes)>>;
    auto ContentPtr = std::shared_ptr<ContentType>(ContentRes);
    return DynamicTensor(ContentPtr);
}

DynamicTensor DynamicTensor::SampleFromOtherGaussian(int Dim, std::vector<int> InputVec, DynamicTensor Mean, DynamicTensor Var,DynamicTensor VarL, int Seed,int DeviceNum)
{
    if(Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto GaussianShape = Mean.Shape();
    for(size_t a = 0;a+1<GaussianShape.size();a++)InputVec.push_back(GaussianShape[a]);
    auto OutputShape = InputVec;
    OutputShape.push_back(Dim);
    InputVec.push_back(1);
    DynamicTensor STDSamples = DynamicTensor::SampleFromStdGaussian(Dim, InputVec, Seed, DeviceNum); //(10,2,3)
    if(VarL.Ops == nullptr)VarL= Var.Cholesky();
    return Mean + (STDSamples%VarL.Transpose(-1,-2)).View(OutputShape);// mean:(2,3), varl:(2,3,3)
}

DynamicTensor DynamicTensor::Inverse()
{
    auto InverseContent = Ops->TensorPointer->Inverse();
    using ContentType = std::remove_pointer_t<std::decay_t<decltype(InverseContent)>>;
    auto ContentPtr = std::shared_ptr<ContentType>(InverseContent);
    return DynamicTensor(ContentPtr);
}

DynamicTensor DynamicTensor::Det_Symmetric(DynamicTensor InputL)
{
    DynamicTensor UnitTensor = DynamicTensor::CreateUnitTensor(InputL.ShapeInt(), InputL.Ops->RequiresGrad, InputL.GetDeviceNum());
    DynamicTensor AllOnes = InputL.Copy();
    AllOnes.Fill(1);
    DynamicTensor DiagRes = InputL*UnitTensor + AllOnes - UnitTensor;
    int ShapeLen = InputL.Shape().size();
    DynamicTensor Res = DiagRes.EleLog().Sum({ShapeLen-2, ShapeLen-1}, true).Eleexp(M_E).Pow(2.);
    return Res;
}

DynamicTensor DynamicTensor::ProbabilityDensity_Gaussian(DynamicTensor InputSample, DynamicTensor InputMean, DynamicTensor InputVarInv, DynamicTensor InputVarDet)
{
    // sample_num:m, gaussian_num:u, dim_num:n
    // InputSample:(m, u, n)
    // InputMean:(u, n)
    // InputVarInv:(u, n, n)
    // InputVarDet:(u, 1, 1)
    auto OutputShape = InputSample.ShapeInt();
    OutputShape.push_back(1);
    DynamicTensor XMinusMean = (InputSample - InputMean).View(OutputShape); //(m, u, n, 1)
    DynamicTensor XMinusMeanT = XMinusMean.Transpose(-1, -2); //(m, u, 1, n)
    DynamicTensor ExpPartial = (XMinusMeanT % InputVarInv % XMinusMean * (-0.5)).Eleexp(M_E); //(m, u, 1, 1)
    DynamicTensor CPartial = InputVarDet.Pow(-0.5)*std::pow(2.*M_PI, -InputSample.ShapeInt().back()*0.5); //(m, u, 1, 1)
    DynamicTensor ProtoRes = ExpPartial*CPartial; //(m, u, 1, 1)
    auto ProtoShape = ProtoRes.ShapeInt(); //(m, u, 1, 1)
    ProtoShape.pop_back(); //(m, u, 1)
    ProtoShape.pop_back(); //(m, u)
    DynamicTensor FinalRes = ProtoRes.View(ProtoShape);//(m, u)
    return FinalRes;
}

DynamicTensor DynamicTensor::ProbabilityDensity_Log_Gaussian(DynamicTensor InputSample, DynamicTensor InputMean, DynamicTensor InputVarInv, DynamicTensor InputVarDet)
{
    // sample_num:m, gaussian_num:u, dim_num:n
    // InputSample:(m, u, n)
    // InputMean:(u, n)
    // InputVarInv:(u, n, n)
    // InputVarDet:(u, 1, 1)
    auto OutputShape = InputSample.ShapeInt();
    OutputShape.push_back(1);
    DynamicTensor XMinusMean = (InputSample - InputMean).View(OutputShape); //(m, u, n, 1)
    DynamicTensor XMinusMeanT = XMinusMean.Transpose(-1, -2); //(m, u, 1, n)
    DynamicTensor LogExpPartial = XMinusMeanT % InputVarInv % XMinusMean * (-0.5); //(m, u, 1, 1)
    DynamicTensor LogCPartial = InputVarDet.EleLog()*(-0.5) - InputSample.ShapeInt().back()*0.5*std::log(2.*M_PI) ; //(m, u, 1, 1)
    DynamicTensor ProtoRes = LogExpPartial + LogCPartial; //(m, u, 1, 1)
    auto ProtoShape = ProtoRes.ShapeInt(); //(m, u, 1, 1)
    ProtoShape.pop_back(); //(m, u, 1)
    ProtoShape.pop_back(); //(m, u)
    DynamicTensor FinalRes = ProtoRes.View(ProtoShape);//(m, u)
    return FinalRes;
}

DynamicTensor DynamicTensor::Max(int InputDim)
{
    if(InputDim < 0)InputDim = Shape().size() + InputDim;
    Tensor* Res = Ops->TensorPointer->Maximum(InputDim);
    
}