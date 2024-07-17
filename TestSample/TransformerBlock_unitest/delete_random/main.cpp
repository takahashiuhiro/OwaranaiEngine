#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
	size_t dvnum = 0;

	Tensor* attn_cattn_weight = new Tensor({3,9},dvnum,{-0.209389716387,0.313938617706,0.539439082146,-0.355767995119,-0.353367090225,-0.415830731392,0.534706830978,-0.217241346836,-0.011811017990,0.100614905357,0.553111672401,-0.346347510815,-0.217413812876,0.553031206131,-0.544376254082,0.515409946442,-0.068495750427,0.028762757778,-0.404539138079,-0.212561607361,0.120650708675,0.121993720531,-0.399396836758,-0.388366460800,0.246734738350,0.340686082840,-0.248414635658});

	Tensor* attn_cattn_bias = new Tensor({1,9},dvnum,{-0.209389716387,0.313938617706,0.539439082146,-0.355767995119,-0.353367090225,-0.415830731392,0.534706830978,-0.217241346836,-0.011811017990});

	Tensor* attn_cproj_weight = new Tensor({3,3},dvnum,{-0.209389716387,0.313938617706,0.539439082146,-0.355767995119,-0.353367090225,-0.415830731392,0.534706830978,-0.217241346836,-0.011811017990});

	Tensor* attn_cproj_bias = new Tensor({1,3},dvnum,{-0.209389716387,0.313938617706,0.539439082146});

	Tensor* mlp_cfc_weight = new Tensor({3,12},dvnum,{-0.485680103302,-0.247496545315,-0.442987531424,0.217600405216,0.273247063160,0.219506263733,-0.026412069798,-0.502533078194,0.560715079308,0.428272485733,-0.426166892052,0.019910454750,-0.227874696255,0.251188397408,0.138770222664,-0.184441655874,0.460518240929,-0.026673555374,-0.278903633356,0.550726890564,-0.012399971485,-0.559777975082,0.311341106892,-0.393450826406,0.241994619370,0.348400592804,0.082190930843,0.361373245716,-0.124685674906,0.189551293850,-0.030301928520,-0.061172962189,-0.451027125120,0.195954144001,0.196026325226,0.253808140755});

	Tensor* mlp_cfc_bias = new Tensor({1,12},dvnum,{-0.485680103302,-0.247496545315,-0.442987531424,0.217600405216,0.273247063160,0.219506263733,-0.026412069798,-0.502533078194,0.560715079308,0.428272485733,-0.426166892052,0.019910454750});

	Tensor* mlp_cproj_weight = new Tensor({12,3},dvnum,{-0.485680103302,-0.247496545315,-0.442987531424,0.217600405216,0.273247063160,0.219506263733,-0.026412069798,-0.502533078194,0.560715079308,0.428272485733,-0.426166892052,0.019910454750,-0.227874696255,0.251188397408,0.138770222664,-0.184441655874,0.460518240929,-0.026673555374,-0.278903633356,0.550726890564,-0.012399971485,-0.559777975082,0.311341106892,-0.393450826406,0.241994619370,0.348400592804,0.082190930843,0.361373245716,-0.124685674906,0.189551293850,-0.030301928520,-0.061172962189,-0.451027125120,0.195954144001,0.196026325226,0.253808140755});

	Tensor* mlp_cproj_bias = new Tensor({1,3},dvnum,{-0.485680103302,-0.247496545315,-0.442987531424});

	he params = he::NewDict();
	params["NHead"] = 3;
	params["NEmbd"] = 3;
	params["Dropout"] = 0.;
	params["BlockSize"] = 40;
	params["Bias"] = 1;
	params["DeviceNum"] = int(dvnum);

	BaseDynamicLayer* ly = new TransformerBlock();
	ly->Init(params);

	//去除随机性
	ly->SubLayers["Attn"]->SubLayers["CAttn"]->Weights["Weight"].Ops->TensorPointer = std::shared_ptr<Tensor>(attn_cattn_weight->Copy());
	ly->SubLayers["Attn"]->SubLayers["CAttn"]->Weights["Bias"].Ops->TensorPointer = std::shared_ptr<Tensor>(attn_cattn_bias->Copy());
	ly->SubLayers["Attn"]->SubLayers["CProj"]->Weights["Weight"].Ops->TensorPointer = std::shared_ptr<Tensor>(attn_cproj_weight->Copy());
	ly->SubLayers["Attn"]->SubLayers["CProj"]->Weights["Bias"].Ops->TensorPointer = std::shared_ptr<Tensor>(attn_cproj_bias->Copy());
	ly->SubLayers["MLP"]->SubLayers["CFC"]->Weights["Weight"].Ops->TensorPointer = std::shared_ptr<Tensor>(mlp_cfc_weight->Copy());
	ly->SubLayers["MLP"]->SubLayers["CFC"]->Weights["Bias"].Ops->TensorPointer = std::shared_ptr<Tensor>(mlp_cfc_bias->Copy());
	ly->SubLayers["MLP"]->SubLayers["CProj"]->Weights["Weight"].Ops->TensorPointer = std::shared_ptr<Tensor>(mlp_cproj_weight->Copy());
	ly->SubLayers["MLP"]->SubLayers["CProj"]->Weights["Bias"].Ops->TensorPointer = std::shared_ptr<Tensor>(mlp_cproj_bias->Copy());

	std::vector<DynamicTensor>inputx = {
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, dvnum, {1,2,1,6,5,0})), 1),
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, dvnum, {1,2,1,6,4,0})), 1)
	};

	std::vector<DynamicTensor>outputx = {
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, dvnum, {1,2,1,6,5,1})), 0),
		DynamicTensor(std::shared_ptr<Tensor>(new Tensor({1,2,3}, dvnum, {1,2,1,6,5,7})), 0)
	};

	auto sgd = Optimizer::CreateSGD(ly->Parameters(), 0.01);

	for(int a=0;a<inputx.size()+1;a++)
	{
		auto res = ly->Forward({inputx[a%2]})[0];
		auto lossres = (res-outputx[a%2]).Pow(2).Sum()*(1./res.Ops->TensorPointer->ShapeCount);
		lossres.Backward();
		sgd.Step();
		print(res);
		//print(lossres);
	}
	//print(ly->Forward({inputx[0]})[0]);
}
