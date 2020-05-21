#pragma once
typedef unsigned char byte;

class Mandelbulb
{
private:
	int n;
	int maxIter;
	float bailout;
	float sqrBailout;
	byte* points = nullptr;
	size_t side = 0;

public:
	Mandelbulb(int power, int maxIter);
	bool compute(size_t width, size_t height);
	void draw(size_t width, size_t height);
};

