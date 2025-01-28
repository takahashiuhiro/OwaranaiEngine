#include <vector>

template<typename GPUDeviceProcessT, typename GLSLT, typename BufferT, typename VBufferT>
void FillRandomValBernoulliInCPPComb(int GLSLFunName ,int WorkNum ,std::vector<BufferT> FunctionParams)
{
    GPUDeviceProcessT::I().ProcessGLSLFun
    (
        GLSLT::I().FillRandomValUniformInCPP, 
        WorkNum,
        {
            FunctionParams[0], 
            FunctionParams[1], 
            VBufferT::CVBuffer(float(0)).OpenGLTMPBuffer,
            VBufferT::CVBuffer(float(1)).OpenGLTMPBuffer,
            FunctionParams[3], 
        }
    );
    GPUDeviceProcessT::I().ProcessGLSLFun
    (
        GLSLT::I().GenerateSignTensorInCPP, 
        WorkNum,
        {
            FunctionParams[0], 
            FunctionParams[1], 
            FunctionParams[2], 
        }
    );
}


template<typename GPUDeviceProcessT, typename GLSLT, typename BufferT, typename VBufferT>
void GPUFunCombFun(int GLSLFunName ,int WorkNum ,std::vector<BufferT> FunctionParams)
{
    if(GLSLFunName == GLSLT::I().FillRandomValBernoulliInCPP)
    {
        FillRandomValBernoulliInCPPComb<GPUDeviceProcessT,GLSLT,BufferT,VBufferT>(GLSLFunName, WorkNum, FunctionParams);
        return;
    }
}