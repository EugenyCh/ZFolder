#include "Hypercomplex.h"
#include <cmath>

Hypercomplex::Hypercomplex() : x(0.0f), y(0.0f), z(0.0f) {}

Hypercomplex::Hypercomplex(float a, float b, float c) : x(a), y(b), z(c) {}

float Hypercomplex::radius()
{
	return sqrtf(x * x + y * y + z * z);
}

float Hypercomplex::sqrRadius()
{
	return x * x + y * y + z * z;
}

float Hypercomplex::phi()
{
	return atan2f(y, x);
}

float Hypercomplex::theta()
{
	return atan2f(sqrtf(x * x + y * y), z);
}

Hypercomplex Hypercomplex::operator+(const Hypercomplex& a)
{
	return Hypercomplex{ x + a.x, y + a.y, z + a.z };
}

Hypercomplex Hypercomplex::operator*(const float a)
{
	return Hypercomplex{ x * a, y * a, z * a };
}

Hypercomplex Hypercomplex::operator^(const float n)
{
	float rn = powf(radius(), n);
	float ph = phi();
	float th = theta();
	return Hypercomplex(
		sinf(n * th) * cosf(n * ph),
		sinf(n * th) * sinf(n * ph),
		cosf(n * th)
	) * rn;
}
