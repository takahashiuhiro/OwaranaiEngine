#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include "Code/AutomaticDifferentiation/AutoDiffCommon.h"
#include <cmath>
#include <fstream>
#include "Code/AutomaticDifferentiation/Layers/BaseLayer.h"
#include "Code/AutomaticDifferentiation/Layers/LinearSoftmaxLayer.h"
int main() 
{
    LinearSoftmaxLayer *m = new LinearSoftmaxLayer(nullptr, "gachi",1,{3,2,1}, 0);
    std::cout<<m->SubLayers["layer_1"]->LayerName<<std::endl;
    std::cout<<m->SubLayers["layer_1"]->DeviceNum<<std::endl;
}
