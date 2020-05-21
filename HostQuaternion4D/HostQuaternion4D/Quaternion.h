#pragma once
#include <cmath>

struct Quaternion
{
	float r, a, b, c;

	Quaternion() : r(0.0f), a(0.0f), b(0.0f), c(0.0f) {}

	Quaternion(float _r, float _a, float _b, float _c) : r(_r), a(_a), b(_b), c(_c) {}

	float radius()
	{
		return sqrtf(r * r + a * a + b * b + c * c);
	}

	float sqrRadius()
	{
		return r * r + a * a + b * b + c * c;
	}

	Quaternion operator+(const Quaternion& q)
	{
		return Quaternion{ r + q.r, a + q.a, b + q.b, c + q.c };
	}

	Quaternion sqr()
	{
		return Quaternion{
			r * r - a * a - b * b - c * c,
			2.0f * r * a,
			2.0f * r * b,
			2.0f * r * c
		};
	}
};
