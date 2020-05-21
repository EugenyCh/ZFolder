#include <stdio.h>
#include <math.h>
#include <GL/glut.h>
#include "LSystem.h"

LSystem::LSystem()
{
	initialString = "F";
	angle = M_PI / 2;
	distScale = 1.0;
}

void LSystem::interpretString(const string& str)
{
	State state;
	stack<State> st;
	Vector3D pos0;

	state.pos = Vector3D(0, 0, 0);
	state.hlu[0] = Vector3D(1, 0, 0);
	state.hlu[1] = Vector3D(0, 1, 0);
	state.hlu[2] = Vector3D(0, 0, 1);
	state.distScale = 1.0;

	int maxLevel = 0;
	int level = 0;
	for (int i = 0; i < str.length(); i++)
	{
		switch (str[i])
		{
		case '[':
			++level;
			break;

		case ']':
			--level;
			break;
		}
		if (level > maxLevel)
			maxLevel = level;
	}

	bounds.addVertex(state.pos);
	for (int i = 0; i < str.length(); i++)
	{
		float k;
		if (maxLevel > 0)
			k = 1.0f * st.size() / maxLevel;
		else
			k = 1.0f;

		switch (str[i])
		{
		case '[': // push state
			st.push(state);
			state.distScale *= distScale;
			break;

		case ']': // pop state
			state = st.top();
			st.pop();
			break;

		case 'F':
			pos0 = state.pos;
			state.pos.x += state.distScale * state.hlu[0].x;
			state.pos.y += state.distScale * state.hlu[1].x;
			state.pos.z += state.distScale * state.hlu[2].x;
			bounds.addVertex(state.pos);
			drawLine(pos0, state.pos, k);
			break;

		case 'f':
			state.pos.x += state.distScale * state.hlu[0].x;
			state.pos.y += state.distScale * state.hlu[1].x;
			state.pos.z += state.distScale * state.hlu[2].x;
			break;

		case '+':
			state.hlu[0] = state.hlu[0] * Matrix3D::rotateZ(angle);
			state.hlu[1] = state.hlu[1] * Matrix3D::rotateZ(angle);
			state.hlu[2] = state.hlu[2] * Matrix3D::rotateZ(angle);
			break;

		case '-':
			state.hlu[0] = state.hlu[0] * Matrix3D::rotateZ(-angle);
			state.hlu[1] = state.hlu[1] * Matrix3D::rotateZ(-angle);
			state.hlu[2] = state.hlu[2] * Matrix3D::rotateZ(-angle);
			break;

		case '&':
			state.hlu[0] = state.hlu[0] * Matrix3D::rotateY(angle);
			state.hlu[1] = state.hlu[1] * Matrix3D::rotateY(angle);
			state.hlu[2] = state.hlu[2] * Matrix3D::rotateY(angle);
			break;

		case '^':
			state.hlu[0] = state.hlu[0] * Matrix3D::rotateY(-angle);
			state.hlu[1] = state.hlu[1] * Matrix3D::rotateY(-angle);
			state.hlu[2] = state.hlu[2] * Matrix3D::rotateY(-angle);
			break;

		case '<':
		case '\\':
			state.hlu[0] = state.hlu[0] * Matrix3D::rotateX(angle);
			state.hlu[1] = state.hlu[1] * Matrix3D::rotateX(angle);
			state.hlu[2] = state.hlu[2] * Matrix3D::rotateX(angle);
			break;

		case '>':
		case '/':
			state.hlu[0] = state.hlu[0] * Matrix3D::rotateX(-angle);
			state.hlu[1] = state.hlu[1] * Matrix3D::rotateX(-angle);
			state.hlu[2] = state.hlu[2] * Matrix3D::rotateX(-angle);
			break;

		case '|':
			state.hlu[0] = state.hlu[0] * Matrix3D::rotateZ(M_PI);
			state.hlu[1] = state.hlu[1] * Matrix3D::rotateZ(M_PI);
			state.hlu[2] = state.hlu[2] * Matrix3D::rotateZ(M_PI);
			break;
		}
	}
}

void LSystem::buildSystem(int numIterations)
{
	bounds.reset();

	currentString = initialString;

	for (int i = 0; i < numIterations; i++)
	{
		currentString = oneStep(currentString);

		printf("%s\n", currentString.c_str());
	}
}

string LSystem::oneStep(const string& in) const
{
	string out;
	map<char, string>::const_iterator it;

	for (int i = 0; i < in.length(); i++)
	{
		it = rules.find(in[i]);

		if (it != rules.end())
			out += it->second;
		else
			out += in[i];
	}

	return out;
}

void LSystem::drawLine(const Vector3D& p1, const Vector3D& p2, float k) const
{
	// k in [0; 1]
	glLineWidth((width1 - width0) * k + width0);
	glBegin(GL_LINES);
	glColor4f(
		(color1[0] - color0[0]) * k + color0[0],
		(color1[1] - color0[1]) * k + color0[1],
		(color1[2] - color0[2]) * k + color0[2],
		(color1[3] - color0[3]) * k + color0[3]
	);
	glVertex3fv(p1);
	glVertex3fv(p2);
	glEnd();
}
