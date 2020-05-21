#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

struct cuComplex
{
	float r, i;

	__device__ cuComplex(float a, float b) : r(a), i(b) {}

	__device__ float sqrMagnitude(void)
	{
		return r * r + i * i;
	}

	__device__ float magnitude(void)
	{
		return sqrtf(r * r + i * i);
	}

	__device__ cuComplex operator*(const cuComplex& a)
	{
		return { r * a.r - i * a.i, i * a.r + r * a.i };
	}

	__device__ cuComplex operator+(const cuComplex& a)
	{
		return { r + a.r, i + a.i };
	}
};