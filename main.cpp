#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicLayers/Linear.h"

int main() 
{
	DynamicTensor x({ 3,3,5 }, 1);
	x.Fill(3.4);

	Linear* layer = new Linear();
	he params = he::NewDict();
	params["DeviceNum"] = 0;
	params["InFeatures"] = 5;
	params["OutFeatures"] = 4;
	params["Bias"] = 0;

	//layer->Init(params);
	layer->CreateNewLayer<Linear>("gg", params);
	print(layer->SubLayers["gg"]->Forward({x})[0]);
}
