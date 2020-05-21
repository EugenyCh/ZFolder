#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

struct Hypercomplex
{
	float x, y, z;

	__device__ Hypercomplex() : x(0.0f), y(0.0f), z(0.0f) {}

	__device__ Hypercomplex(float a, float b, float c) : x(a), y(b), z(c) {}

	__device__ float radius()
	{
		return sqrtf(x * x + y * y + z * z);
	}

	__device__ float sqrRadius()
	{
		return x * x + y * y + z * z;
	}

	__device__ float phi()
	{
		return atan2f(y, x);
	}

	__device__ float theta()
	{
		return atan2f(sqrtf(x * x + y * y), z);
	}

	__device__ Hypercomplex operator+(const Hypercomplex& a)
	{
		return Hypercomplex{ x + a.x, y + a.y, z + a.z };
	}

	__device__ Hypercomplex operator*(const float a)
	{
		return Hypercomplex{ x * a, y * a, z * a };
	}

	__device__ Hypercomplex operator^(const float n)
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
};
