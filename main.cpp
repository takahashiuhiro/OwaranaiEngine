#include "OwaranaiEngine/OwaranaiEngineInclude.h"


int main()
{
    std::cout<<"gpu test"<<std::endl;
    Tensor* q =new Tensor(std::vector<size_t>{2,2,3}, "GPU", 0);
    q->FillArray(1.);
    Tensor* w =new Tensor(std::vector<size_t>{2,3}, "GPU", 0);
    w->FillArray(2.);
    w->SetV(std::vector<size_t>{1,2}, 99.);
    w->SetV(std::vector<size_t>{0,1}, 899.);
    Tensor* e = q->Add(w);
    q->PrintData();
    w->PrintData();
    e->PrintData();
    
    std::cout<<"cpu test"<<std::endl;
    Tensor* qq =new Tensor(std::vector<size_t>{2,2,3});
    qq->FillArray(1.);
    Tensor* wq =new Tensor(std::vector<size_t>{2,3});
    wq->FillArray(2.);
    wq->SetV(std::vector<size_t>{1,2}, 99.);
    wq->SetV(std::vector<size_t>{0,1}, 899.);
    Tensor* eq = qq->Add(wq);
    qq->PrintData();
    wq->PrintData();
    eq->PrintData();
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
