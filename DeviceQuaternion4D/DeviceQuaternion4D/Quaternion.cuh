#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

struct Quaternion
{
	float r, a, b, c;

	__device__ Quaternion() : r(0.0f), a(0.0f), b(0.0f), c(0.0f) {}

	__device__ Quaternion(float _r, float _a, float _b, float _c) : r(_r), a(_a), b(_b), c(_c) {}

	__device__ float radius()
	{
		return sqrtf(r * r + a * a + b * b + c * c);
	}

	__device__ float sqrRadius()
	{
		return r * r + a * a + b * b + c * c;
	}

	__device__ Quaternion operator+(const Quaternion& q)
	{
		return Quaternion{ r + q.r, a + q.a, b + q.b, c + q.c };
	}

	__device__ Quaternion sqr()
	{
		return Quaternion{
			r * r - a * a - b * b - c * c,
			2.0 * r * a,
			2.0 * r * b,
			2.0 * r * c
		};
	}

	__device__ Quaternion operator^(int n)
	{
		if (n < 2)
			return *this;
		else
			--n;
		Quaternion res = *this;
		Quaternion a = *this;
		while (n)
			if (n & 1) {
				res = res * a;
				--n;
			}
			else {
				a = a.sqr();
				n = n >> 1;
			}
		return res;
	}

	__device__ Quaternion operator*(const Quaternion& q)
	{
		return Quaternion{
			r * q.r - a * q.a - b * q.b - c * q.c,
			r * q.a + a * q.r + b * q.c - c * q.b,
			r * q.b - a * q.c + b * q.r + c * q.a,
			r * q.c + a * q.b - b * q.a + c * q.r
		};
	}
};
