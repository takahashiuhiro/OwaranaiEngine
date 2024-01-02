#include <cmath>
#include <fstream>
#include <random>
#include <chrono>

#ifdef CUDA_USEFUL
#include "Cuda/TensorCoreCudaFun.h"
#endif

struct CudaDimVec
{
  size_t Shape[8];
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

    void InitDevicePointerManager(size_t DeviceNum, size_t ShapeCount)
    {
        for(size_t a=0;a<MaxDeviceNum;a++)
        {
            DataPointers.push_back(nullptr);
        }
        SetDevice(DeviceNum, ShapeCount);
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
        }
        else
        {
            bool CudaFlag = 0;
            #ifdef CUDA_USEFUL
            CudaFlag = 1;
            cudaMallocInCPP(&DataPointers[NewDeviceNum], ShapeCount, DeviceNumToCuda(NewDeviceNum));
            #endif
            Log::Assert(CudaFlag, std::string("Use Cuda Branch But..."));
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
            }
        }
        FreeOldDevice(OldDeviceNum);
        this->DeviceNum = NewDeviceNum;
    }
};
