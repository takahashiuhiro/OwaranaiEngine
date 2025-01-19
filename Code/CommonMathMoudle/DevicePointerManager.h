#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "GPUDeviceProcess.h"

#ifdef CUDA_USEFUL
#include "Cuda/TensorCoreCudaFun.h"
#endif

struct CudaDimVec
{
    int ShapeLen = 8;
    size_t Shape[8];
    std::vector<int>ShapeStd;
    int* ToInt()
    {
        ShapeStd.clear();
        ShapeStd.resize(ShapeLen);
        for(int a=0;a<ShapeLen;a++)ShapeStd.push_back(Shape[a]);
        return ShapeStd.data();
    }
};

static size_t DeviceNumToCuda(size_t DeviceNum)
{
    //0是cpu
    return DeviceNum-1;
}

struct DevicePointerManager
{

    DevicePointerManager(size_t DeviceNum, size_t ShapeCount)
    {
        this->MaxDeviceNum = 2;
        InitDevicePointerManager(DeviceNum, ShapeCount);
    }

    DevicePointerManager(size_t MaxDeviceNum, size_t DeviceNum, size_t ShapeCount)
    {
        this->MaxDeviceNum = MaxDeviceNum;
        InitDevicePointerManager(DeviceNum, ShapeCount);
    }

    void InitDevicePointerManager(size_t InputDeviceNum, size_t ShapeCount)
    {
        #ifdef OPENGL_USEFUL
        Log::Assert(InputDeviceNum<=1, "opengl's devicenum must less than 2!");
        #endif
        for(size_t a=0;a<MaxDeviceNum;a++)
        {
            DataPointers.push_back(nullptr);
        }
        SetDevice(InputDeviceNum, ShapeCount);
    }

    ~DevicePointerManager()
    {
        FreeOldDevice(this->DeviceNum);
    }

    std::vector<float*>DataPointers;
    /**最大的设备数.*/
    size_t MaxDeviceNum = 1;
    /**当前的设备数.*/
    size_t DeviceNum = ImpossibleFrameMaxDeviceNum();

    #ifdef OPENGL_USEFUL
    GLuint OpenGLDataPointer;
    #endif

    size_t FrameMaxDeviceNum()
    {
        return 5000;
    }

    size_t ImpossibleFrameMaxDeviceNum()
    {
        return FrameMaxDeviceNum()+1;
    }

    float* GetDevicePointer()
    {
        #ifdef OPENGL_USEFUL
        Log::Assert(DeviceNum==0,"OpenGL has no DevicePointer");
        #endif
        return DataPointers[DeviceNum];
    }

    void FreeOldDevice(size_t OldDeviceNum)
    {
        //释放对应设备的内存
        if(OldDeviceNum == ImpossibleFrameMaxDeviceNum())return;
        if(!DataPointers[OldDeviceNum])return;
        if(!OldDeviceNum)
        {
            free(DataPointers[OldDeviceNum]);
        }
        else
        {
            #ifdef CUDA_USEFUL
            cudaFreeInCPP(DataPointers[OldDeviceNum]);
            #endif
            #ifdef OPENGL_USEFUL
            glDeleteBuffers(1, &OpenGLDataPointer);
            #endif
        }
    }

    void SetDevice(size_t NewDeviceNum, size_t ShapeCount)
    {
        //设置对应设备的内存
        size_t OldDeviceNum = DeviceNum;
        if(NewDeviceNum == OldDeviceNum)return;
        if(!NewDeviceNum)
        {
            DataPointers[NewDeviceNum] = (float*)malloc(sizeof(float)*ShapeCount);
            #ifdef CUDA_USEFUL
            if(OldDeviceNum !=ImpossibleFrameMaxDeviceNum())DataGPUToCPU(DataPointers[NewDeviceNum], DataPointers[OldDeviceNum], ShapeCount);
            #endif
            #ifdef OPENGL_USEFUL
            if(OldDeviceNum !=ImpossibleFrameMaxDeviceNum())GPUDeviceProcess::I().DataGPUToCPU(OpenGLDataPointer,DataPointers[NewDeviceNum],ShapeCount);
            #endif
        }
        else
        {
            size_t GPUFlag = 0;
            #ifdef CUDA_USEFUL
            GPUFlag = 1;
            cudaMallocInCPP(&DataPointers[NewDeviceNum], ShapeCount, DeviceNumToCuda(NewDeviceNum));
            #endif
            #ifdef OPENGL_USEFUL
            GPUFlag = 2;
            OpenGLDataPointer = GPUDeviceProcess::I().GetBuffer_OpenGL(ShapeCount);
            #endif
            Log::Assert(GPUFlag, std::string("Use GPU Branch But...Your -DBACKWARD=CPU..."));
            if(OldDeviceNum !=ImpossibleFrameMaxDeviceNum())
            {
                #ifdef CUDA_USEFUL
                if(!OldDeviceNum)
                {
                    DataCPUToGPU(DataPointers[OldDeviceNum], DataPointers[NewDeviceNum], ShapeCount);
                }
                else
                {
                    DataGPUToGPU(DataPointers[NewDeviceNum], DataPointers[OldDeviceNum], ShapeCount);
                }
                #endif
                #ifdef OPENGL_USEFUL
                if(!OldDeviceNum)
                {
                    GPUDeviceProcess::I().DataCPUToGPU(OpenGLDataPointer, DataPointers[OldDeviceNum], ShapeCount);
                }
                else
                {
                    Log::Assert(false, "OpenGL can not GPU TO GPU");
                }
                #endif
            }
        }
        FreeOldDevice(OldDeviceNum);
        this->DeviceNum = NewDeviceNum;
    }
};
