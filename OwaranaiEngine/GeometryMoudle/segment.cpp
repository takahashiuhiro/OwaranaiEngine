#include"segment.h"

segment::segment(){}


segment segment::init (vector start_vector, vector end_vector)
{
	this->start_vector = start_vector;
	this->end_vector = end_vector;
	this->direction_vector = (end_vector - start_vector).direction_vector();
	return *this;
}

double segment::length(){return (end_vector - start_vector).length();}

vector segment::projection(vector moto_vector)
{
	vector seg_start_to_motovector = moto_vector - this->start_vector;
	double projection_len = seg_start_to_motovector * this->direction_vector;
	return this->start_vector + this->direction_vector * projection_len;
}
