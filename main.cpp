#include "OwaranaiEngine/OwaranaiEngineInclude.h"


int main()
{
    std::cout<<"gpu test"<<std::endl;
    Tensor* q =new Tensor(std::vector<size_t>{1,2,3}, "GPU", 0);
    q->FillArray(1.);
    Tensor* w =new Tensor(std::vector<size_t>{1,2,3}, "GPU", 0);
    w->FillArray(2.);
    Tensor* e = q->AddArray(w);
    e->PrintData();
    Tensor* r = e->AddArray(e);
    r->PrintData();
    std::cout<<"cpu test"<<std::endl;
    Tensor* qq =new Tensor(std::vector<size_t>{3,2,2,3});
    qq->FillArray(1.);
    Tensor* ww =new Tensor(std::vector<size_t>{2,2,3});
    ww->FillArray(0.);
    ww->SetV(std::vector<size_t>{0,0,0}, 5);
    //std::cout<<ww->GetV(std::vector<size_t>{0,0,0})<<std::endl;
    Tensor* ee = qq->Add(ww);
    ee->PrintData();
    //Tensor* rr = ee->AddArray(ee);
    //rr->PrintData();

    //std::vector<int>h = {1,2,3};
    //std::vector<int>hh = std::move(h);
    //for(int a=0;a<h.size();a++)std::cout<<h[a]<<std::endl;
    //for(int a=0;a<hh.size();a++)std::cout<<hh[a]<<std::endl;
    //h = {4,5,6};
    //for(int a=0;a<h.size();a++)std::cout<<h[a]<<std::endl;

    //std::vector<vector> qyy;
	//qyy.push_back(vector(0, 0, 0));
	//qyy.push_back(vector(5, 0, 0));
	//qyy.push_back(vector(5, 5, 0));
	//qyy.push_back(vector(0, 5, 0));
	//qyy.push_back(vector(3, 8, 0));
	//qyy.push_back(vector(3, 3, 0));
	//std::vector<vector>ert;
	//convex_polygon_2d p = convex_polygon_2d(&ert);
	//plane().get_convex_polygon_2d(&qyy, &p);
	//for (int a = 0; a < p.vector_list->size(); a++)(*(p.vector_list))[a].print();
	//vector t = vector(1, 2, 3);
	//t.print();
}
