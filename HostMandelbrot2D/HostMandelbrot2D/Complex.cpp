#include "Complex.h"

Complex::Complex(float a, float b) : r(a), i(b) {}

float Complex::sqrMagnitude(void)
{
	return r * r + i * i;
}

Complex Complex::operator*(const Complex& a)
{
	return Complex{ r * a.r - i * a.i, i * a.r + r * a.i };
}

Complex Complex::operator+(const Complex& a)
{
	return Complex{ r + a.r, i + a.i };
}