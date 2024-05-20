#include "DynamicTensor.h"
#include "../CommonMathMoudle/OpsType.h"

DynamicOps::~DynamicOps()
{
	for (size_t a = 0; a < InputOpsList.size(); a++)
	{
		if (InputOpsList[a]->OutputOpsSet.find(this) != InputOpsList[a]->OutputOpsSet.end())
		{
			InputOpsList[a]->OutputOpsSet.erase(this);
		}
	}
}

DynamicTensor::DynamicTensor()
{
	OpsSetInMap();
};
DynamicTensor::DynamicTensor(std::shared_ptr<Tensor> InputTensorPointer, bool InputRequiresGrad)
{
	Ops = std::make_shared<DynamicOps>();
	Ops->TensorPointer = InputTensorPointer;
	Ops->RequiresGrad = InputRequiresGrad;
	Ops->LeafNode = this;
	OpsSetInMap();
}
DynamicTensor::DynamicTensor(std::shared_ptr<DynamicOps>InputOps)
{
	Ops = InputOps;
	Ops->LeafNode = this;
	OpsSetInMap();
}

DynamicTensor::DynamicTensor(std::vector<size_t>InputShape, bool InputRequiresGrad, size_t DeviceNum)
{
	Ops = std::make_shared<DynamicOps>();
	Ops->TensorPointer = std::make_shared<Tensor>(InputShape, DeviceNum);
	Ops->RequiresGrad = InputRequiresGrad;
	Ops->LeafNode = this;
	OpsSetInMap();
}

DynamicTensor::DynamicTensor(std::vector<size_t>InputShape, std::vector<float>InputData, bool InputRequiresGrad, size_t DeviceNum)
{
	Ops = std::make_shared<DynamicOps>();
	Ops->TensorPointer = std::make_shared<Tensor>(InputShape, DeviceNum, InputData);
	Ops->RequiresGrad = InputRequiresGrad;
	Ops->LeafNode = this;
	OpsSetInMap();
}

DynamicTensor DynamicTensor::CreateVector(std::vector<float>InputData, bool InputRequiresGrad, size_t DeviceNum)
{
	std::vector<size_t>InputShape = { 1,InputData.size() };
	return DynamicTensor(InputShape, InputData, InputRequiresGrad, DeviceNum);
}

DynamicTensor::~DynamicTensor()
{
	if(Ops != nullptr)
	{
		if(Ops->LeafNode == this)Ops->LeafNode = nullptr;
	}
}

DynamicTensor DynamicTensor::Grad()
{
	Log::Assert(Ops->GradOps != nullptr, "This DynamicTensor Has No Grad");
	return DynamicTensor(Ops->GradOps);
}

DynamicTensor DynamicTensor::Copy()
{
	DynamicTensor Res(std::shared_ptr<Tensor>(Ops->TensorPointer->Copy()), 0);
	return Res;
}

void DynamicTensor::PrintData()
{
	Ops->TensorPointer->PrintData();
}

void DynamicTensor::Fill(float InputValue)
{
	Ops->TensorPointer->FillArray(InputValue);
}

void DynamicTensor::FillRandBernoulli(float P, int Seed)
{
	if (Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
	Ops->TensorPointer->FillRandomValBernoulli(P, Seed);
}

void DynamicTensor::FillRandValUniform(float MinV, float MaxV, int Seed)
{
	if (Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
	Ops->TensorPointer->FillRandomValUniform(MinV, MaxV, Seed);
}

DynamicTensor DynamicTensor::SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor>InputList, he InputPrams, size_t InputOpsType, bool RequiresGrad)
{
	bool MaxRequiresGrad = 0,MaxIsEval = 0;
	for (size_t a = 0; a < InputList.size(); a++)MaxRequiresGrad |= InputList[a].Ops->RequiresGrad;
	for (size_t a = 0; a < InputList.size(); a++)MaxIsEval |= InputList[a].Ops->IsEval;
	DynamicTensor Res(std::shared_ptr<Tensor>(ResTensor), MaxRequiresGrad&RequiresGrad&(!MaxIsEval));
	Res.Ops->IsEval = MaxIsEval;
	if ((!RequiresGrad)|| MaxIsEval)return Res;
	Res.Ops->DynamicOpsType = InputOpsType;
	Res.Ops->Params = InputPrams;
	for (size_t a = 0; a < InputList.size(); a++)
	{
		Res.Ops->InputOpsList.push_back(InputList[a].Ops);
		InputList[a].Ops->OutputOpsSet.insert(Res.Ops.get());
	}
	return Res;
}

void DynamicTensor::Backward(DynamicTensor Loss,bool ClearGrad)
{
	Log::Assert(Ops->OutputOpsSet.size() == 0, "DynamicTensor Backward Must Be Output Data");
	std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>BackwardOpsMap;
	std::map<DynamicOps*, std::set<DynamicOps*>>OutputSetSize;
	GetAllOutputSizeBeforeBackward(OutputSetSize, Ops);
	BackwardDFS(BackwardOpsMap, OutputSetSize,Loss, Ops);
	if (ClearGrad)BackwardClearDFS(Ops);
}

void DynamicTensor::BackwardClearDFS(std::shared_ptr<DynamicOps>CurOps)
{
	if (CurOps->InputOpsList.size())CurOps->GradOps = nullptr;
	else
	{
		if(CurOps->RequiresGrad)
		{
			CurOps->GradOps->InputOpsList = {};
			CurOps->GradOps->OutputOpsSet = {};
		}
	}
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)BackwardClearDFS(CurOps->InputOpsList[a]);
}

void DynamicTensor::BackwardDFS(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, DynamicTensor Loss, std::shared_ptr<DynamicOps>CurOps)
{
	if (CheckPartialGradReady(BackwardOpsMap,OutputSetSize, CurOps))
	{
		if (OutputSetSize[CurOps.get()].empty())
		{
			GenEmptyGradDynamicTensor(Loss);
		}
		else
		{
			std::vector<DynamicOps*>OutputList;
			for (auto Item : OutputSetSize[CurOps.get()])OutputList.push_back(Item);
			DynamicTensor ThisOpsGradRes = DynamicTensor(BackwardOpsMap[CurOps.get()][OutputList[0]]);
			for (size_t a = 1; a < OutputList.size(); a++)
			{
				DynamicTensor PartRes = DynamicTensor(BackwardOpsMap[CurOps.get()][OutputList[a]]);
				ThisOpsGradRes = DynamicTensor::DynamicStdOps_Forward_Add({ ThisOpsGradRes, PartRes },he(), true);
			}
			if (CurOps->GradOps != nullptr)ThisOpsGradRes.Ops->TensorPointer = std::shared_ptr<Tensor>(CurOps->GradOps->TensorPointer->Add(ThisOpsGradRes.Ops->TensorPointer.get()));
			CurOps->GradOps = ThisOpsGradRes.Ops;
		}
		for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
		{
			if (BackwardOpsMap.find(CurOps->InputOpsList[a].get()) == BackwardOpsMap.end())
			{
				BackwardOpsMap[CurOps->InputOpsList[a].get()] = {};
			}
		}
		if(BackwardOps.find(CurOps->DynamicOpsType)!=BackwardOps.end())BackwardOps[CurOps->DynamicOpsType](BackwardOpsMap,CurOps);//ͨ通过算子传输partial grad
		for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
		{
			if (!CurOps->InputOpsList[a]->RequiresGrad)continue;//不需要求导的不用dfs
			BackwardDFS(BackwardOpsMap, OutputSetSize, Loss, CurOps->InputOpsList[a]);
		}
	}
	else
	{
		return;
	}
}

bool DynamicTensor::CheckPartialGradReady(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap, std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps)
{
	if (BackwardOpsMap.find(CurOps.get()) == BackwardOpsMap.end())
	{
		BackwardOpsMap[CurOps.get()] = {};
	}
	return BackwardOpsMap[CurOps.get()].size() == OutputSetSize[CurOps.get()].size();
}

void DynamicTensor::GenEmptyGradDynamicTensor(DynamicTensor Loss)
{
	Tensor* GradResTensor;
	if(Loss.Ops == nullptr)
	{
		GradResTensor = Ops->TensorPointer->Copy();
		GradResTensor->FillArray(1.);
	}
	else GradResTensor = Loss.Ops->TensorPointer->Copy();
	DynamicTensor DynamicTensorGrad(std::shared_ptr<Tensor>(GradResTensor), Ops->RequiresGrad);
	Ops->GradOps = DynamicTensorGrad.Ops;
}

void DynamicTensor::GetAllOutputSizeBeforeBackward(std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps)
{
	if (OutputSetSize.find(CurOps.get()) != OutputSetSize.end())return;
	OutputSetSize[CurOps.get()] = CurOps->OutputOpsSet;
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
	{
		GetAllOutputSizeBeforeBackward(OutputSetSize, CurOps->InputOpsList[a]);
	}
}

DynamicTensor DynamicTensor::ViewAndBC(DynamicTensor ThisDT, DynamicTensor Other, DynamicTensor(*InputFun)(std::vector<DynamicTensor>, he, bool), bool IsMatmul)
{
	//View
	auto ViewFun = [](DynamicTensor L, DynamicTensor S, DynamicTensor(*View)(std::vector<DynamicTensor>, he, bool))
		{
			he ViewParams = he::NewDict();
			ViewParams["ViewDims"] = he::NewList();
			size_t ResiShape = L.Ops->TensorPointer->shape.size() - S.Ops->TensorPointer->shape.size();
			for (size_t a = 0; a < ResiShape; a++)ViewParams["ViewDims"].append(1);
			for (size_t a = 0; a < S.Ops->TensorPointer->shape.size(); a++)ViewParams["ViewDims"].append(int(S.Ops->TensorPointer->shape[a]));
			return View({ S }, ViewParams, true);
		};
	if (!IsMatmul)
	{
		if (ThisDT.Ops->TensorPointer->shape.size() < Other.Ops->TensorPointer->shape.size())ThisDT = ViewFun(Other, ThisDT, DynamicStdOps_Forward_View);
		if (ThisDT.Ops->TensorPointer->shape.size() > Other.Ops->TensorPointer->shape.size())Other = ViewFun(ThisDT, Other, DynamicStdOps_Forward_View);
	}
	else
	{
		if (ThisDT.Ops->TensorPointer->shape.size() < Other.Ops->TensorPointer->shape.size())ThisDT = ViewFun(Other, ThisDT, DynamicStdOps_Forward_View);
		else if (ThisDT.Ops->TensorPointer->shape.size() > Other.Ops->TensorPointer->shape.size())
		{
			if (Other.Ops->TensorPointer->shape.size() == 1)
			{
				he ViewParams = he::NewDict();
				ViewParams["ViewDims"] = he::NewList();
				size_t ResiShape = ThisDT.Ops->TensorPointer->shape.size() - Other.Ops->TensorPointer->shape.size() - 1;
				for (size_t a = 0; a < ResiShape; a++)ViewParams["ViewDims"].append(1);
				for (size_t a = 0; a < Other.Ops->TensorPointer->shape.size(); a++)ViewParams["ViewDims"].append(int(Other.Ops->TensorPointer->shape[a]));
				ViewParams["ViewDims"].append(1);
				Other = DynamicStdOps_Forward_View({ Other }, ViewParams, true);
			}
			else Other = ViewFun(ThisDT, Other, DynamicStdOps_Forward_View);
		}
		else
		{
			if (Other.Ops->TensorPointer->shape.size() == 1)
			{
				he ThisViewParams = he::NewDict();
				ThisViewParams["ViewDims"] = he::NewList();
				he OtherViewParams = he::NewDict();
				OtherViewParams["ViewDims"] = he::NewList();
				ThisViewParams["ViewDims"].append(1);
				ThisViewParams["ViewDims"].append(int(ThisDT.Ops->TensorPointer->shape[0]));
				OtherViewParams["ViewDims"].append(int(Other.Ops->TensorPointer->shape[0]));
				OtherViewParams["ViewDims"].append(1);
				ThisDT = DynamicStdOps_Forward_View({ ThisDT }, ThisViewParams, true);
				Other = DynamicStdOps_Forward_View({ Other }, OtherViewParams, true);
			}
		}
		if (Other.Ops->TensorPointer->shape.size() == 2)
		{
			he MatmulParams = he::NewDict();
			MatmulParams["is_input_1st_T"] = false;
			MatmulParams["is_input_2nd_T"] = false;
			return InputFun({ ThisDT, Other }, MatmulParams, true);
		}
	}
	//BC
	he BCParams = he::NewDict();
	BCParams["BroadCastToShape"] = he::NewList();
	int BCFlag = 0;
	for (size_t a = 0; a < ThisDT.Ops->TensorPointer->shape.size() - 2*IsMatmul; a++)
	{
		int ThisShapeNum = ThisDT.Ops->TensorPointer->shape[a];
		int OtherShapeNum = Other.Ops->TensorPointer->shape[a];
		Log::Assert(!(ThisShapeNum != OtherShapeNum && std::min(ThisShapeNum, OtherShapeNum) != 1), "DynamicTensor Opeator+ Shape Error, Dim Can Not BroadCast");
		BCParams["BroadCastToShape"].append(std::max(ThisShapeNum, OtherShapeNum));
		if (ThisShapeNum < OtherShapeNum)BCFlag |= 1;
		if (ThisShapeNum > OtherShapeNum)BCFlag |= 2;
	}
	if (!IsMatmul)
	{
		if (!BCFlag)return InputFun({ ThisDT, Other }, he(), true);
		if (BCFlag == 1)return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), Other }, he(), true);
		if (BCFlag == 2)return InputFun({ ThisDT, DynamicStdOps_Forward_BroadCastTo({Other}, BCParams,true) }, he(), true);
		return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), DynamicStdOps_Forward_BroadCastTo({Other},BCParams,true) }, he(), true);
	}
	else
	{
		he MatmulParams = he::NewDict();
		MatmulParams["is_input_1st_T"] = false;
		MatmulParams["is_input_2nd_T"] = false;
		if (!BCFlag)return InputFun({ ThisDT, Other }, MatmulParams, true);
		if (BCFlag == 1)
		{
			BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 2]));
			BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 1]));
			return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), Other }, MatmulParams, true);
		}
		if (BCFlag == 2)
		{
			BCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 2]));
			BCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 1]));
			return InputFun({ ThisDT, DynamicStdOps_Forward_BroadCastTo({Other}, BCParams,true) }, MatmulParams, true);
		}
		he OtherBCParams = BCParams;
		BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 2]));
		BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 1]));
		OtherBCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 2]));
		OtherBCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 1]));
		return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), DynamicStdOps_Forward_BroadCastTo({Other},OtherBCParams,true) }, MatmulParams, true);
	}
}

DynamicTensor DynamicTensor::operator+(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())return ViewAndBC(*this, Other, DynamicStdOps_Forward_Add, false);
	else return DynamicStdOps_Forward_Add({ *this, DynamicStdOps_Forward_Add({Other},he(),true) }, he(), true);
}
DynamicTensor DynamicTensor::operator+(float Other)
{
	DynamicTensor Scalar({ 1 }, 0, Ops->TensorPointer->GetDeviceNum());
	Scalar.Fill(Other);
	return operator+(Scalar);
}

DynamicTensor DynamicTensor::operator%(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())return ViewAndBC(*this, Other, DynamicStdOps_Forward_Matmul, true);
	else
	{
		he MatmulParams = he::NewDict();
		MatmulParams["is_input_1st_T"] = false;
		MatmulParams["is_input_2nd_T"] = false;
		return DynamicStdOps_Forward_Matmul({ *this, DynamicStdOps_Forward_Add({Other},he(),true) }, MatmulParams, true);
	}
}
DynamicTensor DynamicTensor::operator*(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())return ViewAndBC(*this, Other, DynamicStdOps_Forward_Elemul, false);
	else return DynamicStdOps_Forward_Add({ *this, DynamicStdOps_Forward_Elemul({Other},he(),true) }, he(), true);
}
DynamicTensor DynamicTensor::operator*(float Other)
{
	DynamicTensor Scalar({ 1 }, 0, Ops->TensorPointer->GetDeviceNum());
	Scalar.Fill(Other);
	return operator*(Scalar);
}
DynamicTensor DynamicTensor::operator-(DynamicTensor Other)
{
	DynamicTensor MinusOne({1}, 0, Other.Ops->TensorPointer->GetDeviceNum());
	MinusOne.Fill(-1);
	return operator+(MinusOne * Other);
}
DynamicTensor DynamicTensor::operator-(float Other)
{
	DynamicTensor Scalar({ 1 }, 0, Ops->TensorPointer->GetDeviceNum());
	Scalar.Fill(Other);
	return operator-(Scalar);
}