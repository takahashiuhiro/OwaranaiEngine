#include "OwaranaiEngine/OwaranaiEngineInclude.h"
int main()
{
    int q = 1;
    int& w = q;
    std::cout<<&w<<std::endl;
    std::cout<<&q<<std::endl;
    std::cout<<1;
}
