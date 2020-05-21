#pragma once

class Complex
{
public:
	float r, i;

	Complex(float a, float b);
	float sqrMagnitude(void);
	Complex operator*(const Complex& a);
	Complex operator+(const Complex& a);
};

