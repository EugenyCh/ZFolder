#pragma once

class Hypercomplex
{
public:
	float x, y, z;

	Hypercomplex();
	Hypercomplex(float a, float b, float c);
	float radius();
	float sqrRadius();
	float phi();
	float theta();
	Hypercomplex operator+(const Hypercomplex& a);
	Hypercomplex operator*(const float a);
	Hypercomplex operator^(const float n);
};
