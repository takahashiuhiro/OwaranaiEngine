#include "DynamicTensor.h"

void DynamicTensor::OpsSetInMap()
{
	BackwardOps[OpsType::Add] = DynamicStdOps_Backward_Add;
	BackwardOps[OpsType::MatMul] = DynamicStdOps_Backward_Matmul;
	BackwardOps[OpsType::Sum] = DynamicStdOps_Backward_Sum;
	BackwardOps[OpsType::BroadCastTo] = DynamicStdOps_Backward_BroadCastTo;
	BackwardOps[OpsType::View] = DynamicStdOps_Backward_View;
	BackwardOps[OpsType::EleMul] = DynamicStdOps_Backward_Elemul;
	BackwardOps[OpsType::Softmax] = DynamicStdOps_Backward_Softmax;
	BackwardOps[OpsType::Pow] = DynamicStdOps_Backward_Pow;
	BackwardOps[OpsType::EleExp] = DynamicStdOps_Backward_Eleexp;
}


DynamicTensor DynamicTensor::DynamicStdOps_Forward_Add(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Copy();
	for (size_t a = 1; a < InputList.size(); a++)
	{
		Tensor* TMPTensor = ResTensorContent->Add(InputList[a].Ops->TensorPointer.get());
		delete ResTensorContent;
		ResTensorContent = TMPTensor;
	}
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::Add, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Add(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
	{
		if (!CurOps->InputOpsList[a]->RequiresGrad)continue;
		auto AddRes = DynamicTensor::DynamicStdOps_Forward_Add({ DynamicTensor(CurOps->GradOps) }, he(), true);
		BackwardOpsMap[CurOps->InputOpsList[a].get()][CurOps.get()] = AddRes.Ops;
	}
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Matmul(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	bool IsInput1stT = InputParams["is_input_1st_T"].i();
	bool IsInput2ndT = InputParams["is_input_2nd_T"].i();
	Tensor* Input1stRes, * Input2ndRes;
	if (IsInput1stT)Input1stRes = InputList[0].Ops->TensorPointer->T();
	else Input1stRes = InputList[0].Ops->TensorPointer.get();
	if (IsInput2ndT)Input2ndRes = InputList[1].Ops->TensorPointer->T();
	else Input2ndRes = InputList[1].Ops->TensorPointer.get();
	auto TensorResult = Input1stRes->Matmul(Input2ndRes);
	if (IsInput1stT)delete Input1stRes;
	if (IsInput2ndT)delete Input2ndRes;
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::MatMul, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Matmul(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap,std::shared_ptr<DynamicOps>CurOps)
{
	bool IsInput1stT = CurOps->Params["is_input_1st_T"].i();
	bool IsInput2ndT = CurOps->Params["is_input_2nd_T"].i();
	if (CurOps->InputOpsList[0]->RequiresGrad)
	{
		he GradParams = he::NewDict();
		DynamicTensor TensorRes;
		if (IsInput1stT == false && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[1]) }, GradParams, true);
		}
		if (IsInput1stT == false && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[1]) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[1]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[1]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = TensorRes.Ops;
	}
	if (CurOps->InputOpsList[1]->RequiresGrad)
	{
		he GradParams = he::NewDict();
		DynamicTensor TensorRes;
		if (IsInput1stT == false && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[0]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		if (IsInput1stT == false && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[0]) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[0]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[0]) }, GradParams, true);
		}
		BackwardOpsMap[CurOps->InputOpsList[1].get()][CurOps.get()] = TensorRes.Ops;
	}
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_BroadCastTo(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	std::vector<size_t>BroadCastToShape;
	for (he a = 0; a < InputParams["BroadCastToShape"].size(); a = a + 1)BroadCastToShape.push_back(InputParams["BroadCastToShape"][a].i());
	auto TensorResult = InputList[0].Ops->TensorPointer->BroadCastTo(BroadCastToShape);
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::BroadCastTo, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_BroadCastTo(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	std::vector<size_t>BroadCastToShape;
	for (he a = 0; a < CurOps->Params["BroadCastToShape"].size(); a = a + 1)BroadCastToShape.push_back(CurOps->Params["BroadCastToShape"][a].i());
	he InputParams = he::NewDict();
	InputParams["SumDims"] = he::NewList();
	for (size_t a = 0; a < CurOps->TensorPointer->shape.size(); a++)
	{
		if (CurOps->TensorPointer->shape[a] != CurOps->InputOpsList[0]->TensorPointer->shape[a])
		{
			InputParams["SumDims"].append(int(a));
		}
	}
	DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_Sum({ DynamicTensor(CurOps->GradOps) }, InputParams, true);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Sum(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	std::vector<size_t>SumDims;
	for (he a = 0; a < InputParams["SumDims"].size(); a = a + 1)SumDims.push_back(InputParams["SumDims"][a].i());
	auto TensorResult = InputList[0].Ops->TensorPointer->Sum(SumDims);
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::Sum, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Sum(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	he InputParams = he::NewDict();
	InputParams["BroadCastToShape"] = he::NewList();
	for (size_t a = 0; a < CurOps->InputOpsList[0]->TensorPointer->shape.size(); a++)InputParams["BroadCastToShape"].append(int(CurOps->InputOpsList[0]->TensorPointer->shape[a]));
	DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_BroadCastTo({ DynamicTensor(CurOps->GradOps) }, InputParams, true);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_View(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	std::vector<size_t>ViewDims;
	int MinusIndex = -1;
	for (he a = 0; a < InputParams["ViewDims"].size(); a = a + 1)
	{
		if (InputParams["ViewDims"][a] < 0)
		{
			MinusIndex = a.i();
			ViewDims.push_back(0);
		}
		else ViewDims.push_back(InputParams["ViewDims"][a].i());
	}
	auto TensorResult = InputList[0].Ops->TensorPointer->View(ViewDims,MinusIndex);
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::View, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_View(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	he InputParams = he::NewDict();
	InputParams["ViewDims"] = he::NewList();
	for (size_t a = 0; a < CurOps->InputOpsList[0]->TensorPointer->shape.size(); a++)InputParams["ViewDims"].append(int(CurOps->InputOpsList[0]->TensorPointer->shape[a]));
	DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_View({ DynamicTensor(CurOps->GradOps) }, InputParams, true);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Elemul(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0].Ops->TensorPointer->EleMul(InputList[1].Ops->TensorPointer.get());
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::EleMul, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Elemul(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (CurOps->InputOpsList[0]->RequiresGrad)
	{
		DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_Elemul({ DynamicTensor(CurOps->GradOps), DynamicTensor(CurOps->InputOpsList[1]) }, he(), true);
		BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
	}
	if (CurOps->InputOpsList[1]->RequiresGrad)
	{
		DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_Elemul({ DynamicTensor(CurOps->GradOps), DynamicTensor(CurOps->InputOpsList[0]) }, he(), true);
		BackwardOpsMap[CurOps->InputOpsList[1].get()][CurOps.get()] = DynamicTensorRes.Ops;
	}
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Softmax(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	int SoftmaxDim = InputParams["SoftmaxDim"].i();
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Softmax(SoftmaxDim);
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::Softmax, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Softmax(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	/**
	return DynamicTensor::D = {d_i, i \in [1,n]}
	any d_i = I_i*(O_i - I \dot O)
	*/
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	int SoftmaxDim = CurOps->Params["SoftmaxDim"].i();
	DynamicTensor MinusRes = (DynamicTensor(CurOps) * DynamicTensor(CurOps->GradOps)).Sum({ SoftmaxDim },true);
	DynamicTensor Res = DynamicTensor(CurOps) * (DynamicTensor(CurOps->GradOps) - MinusRes);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Pow(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	float EleExponent = InputParams["EleExponent"].f();
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Pow(EleExponent);
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::Pow, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Pow(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	float CurEleExponent = CurOps->Params["EleExponent"].f();
	he PowParams = he::NewDict();
	PowParams["EleExponent"] = CurEleExponent - 1;
	DynamicTensor Res = DynamicStdOps_Forward_Pow({ DynamicTensor(CurOps->InputOpsList[0])}, PowParams, true);
	Res = DynamicTensor(CurOps->GradOps)*Res * CurEleExponent;
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Eleexp(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	float EleBaseNum = InputParams["EleBaseNum"].f();
	auto ResTensorContent = InputList[0].Ops->TensorPointer->EleExp(EleBaseNum);
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::EleExp, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Eleexp(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	float EleBaseNum = CurOps->Params["EleBaseNum"].f();
	DynamicTensor Res = DynamicTensor(CurOps) * DynamicTensor(CurOps->GradOps) * std::log(EleBaseNum);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}