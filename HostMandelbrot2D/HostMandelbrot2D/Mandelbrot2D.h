#pragma once
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef unsigned char byte;

class Mandelbrot2D
{
private:
	byte* points = nullptr;
	int width = 0;
	int height = 0;
	float setScalling;

public:
	bool compute(size_t width, size_t height, int iters, float setScalling);
	void draw();
};

