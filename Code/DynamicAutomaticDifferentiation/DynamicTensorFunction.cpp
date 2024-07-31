#include "DynamicTensor.h"

DynamicTensor DynamicTensor::Sum(std::vector<int>Dims, bool KeepDim)
{
	if (Dims.size() == 0)
	{
		for (size_t a = 0; a < Ops->TensorPointer->shape.size(); a++)
		{
			Dims.push_back(a);
		}
	}
	he SumParams = he::NewDict();
	SumParams["SumDims"] = he::NewList();
	for (size_t a = 0; a < Dims.size(); a++)SumParams["SumDims"].append(Dims[a]);
	DynamicTensor Res = DynamicStdOps_Forward_Sum({ *this }, SumParams, true);
	if (KeepDim)return Res;
	std::map<int, int>DimsMp;
	for (size_t a = 0; a < Dims.size(); a++)DimsMp[Dims[a]] = 1;
	he ViewParams = he::NewDict();
	ViewParams["ViewDims"] = he::NewList();
	for (size_t a = 0; a < Res.Ops->TensorPointer->shape.size(); a++)
	{
		if (DimsMp.find(int(a)) == DimsMp.end())ViewParams["ViewDims"].append(int(Res.Ops->TensorPointer->shape[a]));
		else continue;
	}
	return DynamicStdOps_Forward_View({ Res }, ViewParams, true);
}

DynamicTensor DynamicTensor::View(std::vector<int>Dims)
{
	he ViewParams = he::NewDict();
	ViewParams["ViewDims"] = he::NewList();
	for (size_t a = 0; a < Dims.size(); a++)ViewParams["ViewDims"].append(Dims[a]);
	return DynamicTensor::DynamicStdOps_Forward_View({ *this }, ViewParams, true);
}

DynamicTensor DynamicTensor::Softmax(int InputDim)
{
	he SoftmaxParams = he::NewDict();
	if(InputDim<0)InputDim = Shape().size()+InputDim;
	SoftmaxParams["SoftmaxDim"] = InputDim;
	return DynamicStdOps_Forward_Softmax({ *this }, SoftmaxParams, true);
}

DynamicTensor DynamicTensor::Pow(float EleExponent)
{
	he PowParams = he::NewDict();
	PowParams["EleExponent"] = EleExponent;
	return DynamicStdOps_Forward_Pow({ *this }, PowParams, true);
}

DynamicTensor DynamicTensor::Dropout(DynamicTensor Input, float P, bool InPlace)
{
	if (Input.Ops->IsEval)return Input;
	auto DropoutTensor = Input.Ops->TensorPointer->Copy();
	DropoutTensor->FillRandomValBernoulli(1-P);
	Tensor* DropoutTensorDotP = DropoutTensor->MulScalar(1 / (1-P));
	delete DropoutTensor;
	Input.Ops->TensorPointer = std::shared_ptr<Tensor>(DropoutTensorDotP->EleMul(Input.Ops->TensorPointer.get()));
	delete DropoutTensorDotP;
	return Input;
}

std::vector<DynamicTensor> DynamicTensor::Split(int SplitSize, int Dim)
{
	std::vector<int>SplitSections;
	int ProtoDimSize = Ops->TensorPointer->shape[Dim];
	while(ProtoDimSize)
	{
		if(ProtoDimSize > SplitSize)
		{
			SplitSections.push_back(SplitSize);
			ProtoDimSize -= SplitSize;
		}
		else
		{
			SplitSections.push_back(ProtoDimSize);
			ProtoDimSize = 0;
		}
	}
	return Split(SplitSections, Dim);
}
std::vector<DynamicTensor> DynamicTensor::Split(std::vector<int> SplitSections, int Dim)
{
	auto GenLeftMul = Ops->TensorPointer->GenerateSplitTensor(SplitSections, Dim);
	std::vector<DynamicTensor>Res;
	int PreDims = 1;
	int LastDims = 1;
	for (size_t a = 0; a < Ops->TensorPointer->shape.size(); a++)
	{
		if (a < Dim)PreDims *= Ops->TensorPointer->shape[a];
		else LastDims *= Ops->TensorPointer->shape[a];
	}
	DynamicTensor ViewTensor = View({ PreDims , LastDims });
	for (size_t a = 0; a < GenLeftMul.size(); a++)
	{
		DynamicTensor ResTMPTensor = ViewTensor % DynamicTensor(std::shared_ptr<Tensor>(GenLeftMul[a]));
		std::vector<int> ReturnShape;
		for (size_t b = 0; b < Ops->TensorPointer->shape.size(); b++)
		{
			if (b != Dim)ReturnShape.push_back(Ops->TensorPointer->shape[b]);
			else ReturnShape.push_back(SplitSections[a]);
		}
		Res.push_back(ResTMPTensor.View(ReturnShape));
	}
	return Res;
}

DynamicTensor DynamicTensor::Eleexp(float EleBaseNum)
{
	he EleexpParams = he::NewDict();
	EleexpParams["EleBaseNum"] = EleBaseNum;
	return DynamicStdOps_Forward_Eleexp({ *this }, EleexpParams, true);
}

DynamicTensor DynamicTensor::Tanh()
{
	DynamicTensor ExpTMP =  (DynamicTensor(Ops) * (-2)).Eleexp(M_E);
	return (ExpTMP * (-1) + 1) * ((ExpTMP + 1.).Pow(-1.));
}

DynamicTensor DynamicTensor::Cat(std::vector<DynamicTensor>InputTensors, int Dim)
{
	std::vector<Tensor*>TensorVer;
	for (auto& It : InputTensors)TensorVer.push_back(It.Ops->TensorPointer.get());
	auto GenRightMat = Tensor::GenerateCatTensor(TensorVer, Dim);
	DynamicTensor Res;
	int TargetDim = 0;
	for (size_t a = 0; a < InputTensors.size(); a++)TargetDim += InputTensors[a].Ops->TensorPointer->shape[Dim];
	for (size_t a = 0; a < InputTensors.size(); a++)
	{
		size_t FirstDim = 1;
		size_t LastDim = 1;
		for (size_t b = 0; b < InputTensors[a].Ops->TensorPointer->shape.size(); b++)
		{
			if (b < Dim)FirstDim *= InputTensors[a].Ops->TensorPointer->shape[b];
			else LastDim *= InputTensors[a].Ops->TensorPointer->shape[b];
		}
		if (a == 0)Res = InputTensors[a].View({ (int)FirstDim, (int)LastDim }) % DynamicTensor(std::shared_ptr<Tensor>(GenRightMat[a]));
		else Res = Res + InputTensors[a].View({ (int)FirstDim, (int)LastDim }) % DynamicTensor(std::shared_ptr<Tensor>(GenRightMat[a]));
	}
	std::vector<int> ReturnShape;
	for (size_t a = 0; a < InputTensors[0].Ops->TensorPointer->shape.size(); a++)ReturnShape.push_back(InputTensors[0].Ops->TensorPointer->shape[a]);
	ReturnShape[Dim] = TargetDim;
	return Res.View(ReturnShape);
}

DynamicTensor DynamicTensor::GELU()
{
	auto Self = DynamicTensor(Ops);
	return (Self * 0.5) * (((Self + Self.Pow(3.) * 0.044715) * std::pow(2. / M_PI, 0.5)).Tanh() + 1);
}

DynamicTensor DynamicTensor::Mean(std::vector<int>InputDims, bool KeepDim)
{
	float MeanPartial = 1;
	for (size_t a = 0; a < InputDims.size(); a++)MeanPartial *= Ops->TensorPointer->shape[InputDims[a]];
	return Sum(InputDims, KeepDim)*(1./ MeanPartial);
}

DynamicTensor DynamicTensor::Var(std::vector<int>InputDims, bool KeepDim, float Correction)
{
	auto Self = DynamicTensor(Ops);
	float SumDimRes = 1;
	for (size_t a = 0; a < InputDims.size(); a++)SumDimRes *= Ops->TensorPointer->shape[InputDims[a]];
	return (Self - Self.Mean(InputDims, true)).Pow(2.).Sum(InputDims, KeepDim) * (1. / (SumDimRes - Correction));
}

DynamicTensor DynamicTensor::Tril(int Diagonal)
{
	auto Self = DynamicTensor(Ops);
	Tensor* TensorTrilOnes = Tensor::GenerateTrilOnes(Shape(), Diagonal, Ops->TensorPointer->GetDeviceNum());
	return Self*DynamicTensor(std::shared_ptr<Tensor>(TensorTrilOnes));
}

DynamicTensor DynamicTensor::Transpose(int Dim0, int Dim1, int DebugFlag)
{
	he TranposeParams = he::NewDict();
	if(Dim0<0)Dim0 = Ops->TensorPointer->shape.size()+Dim0;
	if(Dim1<0)Dim1 = Ops->TensorPointer->shape.size()+Dim1;
	TranposeParams["Dim0"] = Dim0;
	TranposeParams["Dim1"] = Dim1;
	return DynamicStdOps_Forward_Transpose({*this}, TranposeParams, true);
}

DynamicTensor DynamicTensor::MaskedFill(DynamicTensor Mask, float Value)
{
	DynamicTensor Ones({1}, 0, Mask.Ops->TensorPointer->GetDeviceNum());
	Ones.Fill(1);
	return DynamicTensor(Ops)*(Ones - Mask)+Mask*Value;
}

DynamicTensor DynamicTensor::EleLog()
{
	auto Self = DynamicTensor(Ops);
	return DynamicStdOps_Forward_EleLog({Self}, he(), true);
}

DynamicTensor DynamicTensor::CrossEntropy(DynamicTensor Input, DynamicTensor Target, std::string Reduction, DynamicTensor Weight, float LabelSmoothing)
{
	if(Weight.Ops == nullptr)
	{
		Weight = DynamicTensor(Input.Shape(), false ,Input.GetDeviceNum());
		Weight.Fill(1.);
	}
	auto ExpTensor = Input.Eleexp(M_E);
	auto ExpTensorSum = ExpTensor.Sum({1}, true).Pow(-1);
	auto MiniBatchRes = (ExpTensor*ExpTensorSum).EleLog()*Target*Weight;
	auto MiniBatchResSum = MiniBatchRes.Sum()*(-1);
	if(Reduction == "Sum")return MiniBatchResSum;
	else return MiniBatchResSum*(1./Input.Shape()[0]);
}