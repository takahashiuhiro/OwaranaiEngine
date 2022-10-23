#include "plane.h"

plane::plane(){}


plane::plane(vector fixed_vector, vector normal_vector)
{
	this->fixed_vector = fixed_vector;
	this->normal_vector = normal_vector.direction_vector();

	vector random_vector[3] = { vector(0,0,0), vector(1,1,1), vector(0,0,1)};

	vector second_fixed_vector;
	for (int a = 0; a < 3; a++)
	{
		vector random_check = random_vector[a] - this->fixed_vector;
		double random_check_test = random_check.cross_product(this->normal_vector).length();
		if (random_check_test * random_check_test > 1e-9)
		{
			second_fixed_vector = projection(random_vector[a]);
			break;
		}
	}

	vector tmp = (second_fixed_vector - fixed_vector).direction_vector();
	this->unit_vector[0] = tmp;
	this->unit_vector[1] = this->normal_vector.cross_product(tmp);
	this->unit_vector[0] = this->unit_vector[0].direction_vector();
	this->unit_vector[1] = this->unit_vector[1].direction_vector();
}

plane::plane(vector fixed_vector, vector unit_vector[2])
{
	this->unit_vector[0] = unit_vector[0];
	this->unit_vector[1] = unit_vector[1];
	this->fixed_vector = fixed_vector;
	this->normal_vector = (this->unit_vector[0].cross_product(this->unit_vector[1])).direction_vector();
}

double plane::vector_to_plane_dis(vector vec){return (vec - fixed_vector) * normal_vector;}

vector plane::projection(vector moto_vector){return moto_vector - normal_vector * vector_to_plane_dis(moto_vector);}

vector plane::coordinate_convert_to_plane(vector moto_vector)
{
	vector link_vector = moto_vector - fixed_vector;
	double size = unit_vector[0].x * unit_vector[1].y - unit_vector[0].y * unit_vector[1].x;
	double new_x = link_vector.x * unit_vector[1].y - unit_vector[1].x * link_vector.y;
	double new_y = unit_vector[0].x * link_vector.y - link_vector.x * unit_vector[0].y;
	return vector(new_x/size, new_y / size, 0);
}

vector plane::coordinate_convert_to_std_space(vector plane_vector){return unit_vector[0] * plane_vector.x + unit_vector[1] * plane_vector.y + fixed_vector;}

bool plane::segment_cross(segment seg_1, segment seg_2, vector& cross_vector)
{
	vector seg_1_st_to_seg2 = seg_2.projection(seg_1.start_vector);
	vector seg_1_ed_to_seg2 = seg_2.projection(seg_1.end_vector);
	vector seg_1_st = seg_1.start_vector - seg_1_st_to_seg2;
	vector seg_1_ed = seg_1.end_vector - seg_1_ed_to_seg2;
	if (seg_1_st * seg_1_ed > 1e-9) return 0;

	vector seg_2_st_to_seg1 = seg_1.projection(seg_2.start_vector);
	vector seg_2_ed_to_seg1 = seg_1.projection(seg_2.end_vector);
	vector seg_2_st = seg_2.start_vector - seg_2_st_to_seg1;
	vector seg_2_ed = seg_2.end_vector - seg_2_ed_to_seg1;
	if (seg_2_st * seg_2_ed > 1e-9) return 0;

	double denominator = seg_2_st.length() + seg_2_ed.length();
	if (denominator < 1e-9)
	{
		vector tmp0 = seg_1.start_vector - seg_2_st_to_seg1;
		vector tmp1 = seg_1.end_vector - seg_2_st_to_seg1;
		if (tmp0 * tmp1 < 1e-9) cross_vector = seg_2_st_to_seg1;
		else cross_vector = seg_2_ed_to_seg1;
	}
	else cross_vector = seg_2_st_to_seg1 * (seg_2_ed.length() / denominator) + seg_2_ed_to_seg1 * (seg_2_st.length() / denominator);
	return 1;
}

bool plane::get_convex_polygon_2d(const std::vector<vector>* vector_set_input, convex_polygon_2d* result_polygon)
{
	if (vector_set_input->size() < 3)return 0;
	std::vector<vector>* vector_set = new std::vector<vector>;
	for (int a = 0; a < vector_set_input->size(); a++)vector_set->push_back((*vector_set_input)[a]);

	std::sort(vector_set->begin(), vector_set->end(), vector().vector_compare_x_up);

	int* monoton_stack = (int*)malloc(sizeof(int)* (vector_set->size()+10));
	for (int a = 0; a < vector_set->size(); a++)monoton_stack[a] = 0;

	bool* used = (bool*)malloc(sizeof(bool) *(10+ vector_set->size()));
	for (int a = 0; a < vector_set->size(); a++)used[a] = 0;

	int stack_size = 1;
	
	for (int a = 1; a < vector_set->size(); a++)
	{
		while (stack_size >= 2 && ((*vector_set)[monoton_stack[stack_size - 1]] - (*vector_set)[monoton_stack[stack_size - 2]]).cross_product(((*vector_set)[a] - (*vector_set)[monoton_stack[stack_size - 1]])).z <= 0)
		{
			used[monoton_stack[stack_size - 1]] = 0;
			stack_size -= 1;
		}
		monoton_stack[stack_size++] = a;
		used[a] = 1;
	}

	int one_side_point_num = stack_size;
	for (int a = vector_set->size() - 1; a >= 0; a--)
	{
		if (used[a])continue;
		//std::cout << a<<" "<< stack_size << std::endl;
		while (stack_size > one_side_point_num && ((*vector_set)[monoton_stack[stack_size - 1]] - (*vector_set)[monoton_stack[stack_size - 2]]).cross_product(((*vector_set)[a] - (*vector_set)[monoton_stack[stack_size - 1]])).z <= 0)
		{
			used[monoton_stack[stack_size - 1]] = 0;
			stack_size -= 1;
		}
		monoton_stack[stack_size++] = a;
		used[a] = 1;
	}
	for (int a = 0; a < stack_size; a++)
	{
		result_polygon->vector_list->push_back((*vector_set)[monoton_stack[a]]);
	}
	free(monoton_stack);
	free(used);
	return 1;
}