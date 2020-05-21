#pragma once

#include "Vector3D.h"
#include "Matrix3D.h"
#include "BoundingBox.h"

#include <string>
#include <stack>
#include <map>

using namespace std;

class LSystem
{
    struct State
    {
		Vector3D pos;
		Vector3D hlu[3];
        float distScale;
    };

    map<char, string> rules;
	string initialString;
	float distScale;
	float angle;
	string currentString;
    BoundingBox bounds;

public:
	LSystem();

	float width0;
	float width1;
	float color0[4];
	float color1[4];

    void setInitialString(const char *str)
    {
        initialString = str;
    }

    void addRule(char symbol, const char *rule)
    {
        rules[symbol] = rule;
    }

    void setAngle(float newAngle)
    {
        angle = newAngle;
    }

    const string &getCurrentString() const
    {
        return currentString;
    }

    void setDistScale(float newScale)
    {
        distScale = newScale;
    }

    void interpretString(const string &str);
    void buildSystem(int numIterations);

    void draw()
    {
        interpretString(currentString);
    }

    const BoundingBox &getBounds() const
    {
        return bounds;
    }

protected:
    string oneStep(const string &in) const;
    virtual void drawLine(const Vector3D &p1, const Vector3D &p2, float k) const;
};