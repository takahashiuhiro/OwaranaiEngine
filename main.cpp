#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{ 

    float *i;
    std::vector<float>ddfgl = {1,4,3,1,0,0,4,1,588,0,1,0,3,588,19,0,0,1};
    i = new float[18];
    for(int a=0;a<ddfgl.size();a++)i[a] = ddfgl[a];
    MatrixGaussianElimination(i,3,6);
    for(int a=0;a<ddfgl.size();a++)
    {
        std::cout<<i[a]<<" ";
    }
    std::cout<<std::endl;
}
