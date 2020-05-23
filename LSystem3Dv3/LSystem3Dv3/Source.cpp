#define _CRT_SECURE_NO_WARNINGS
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <FreeImage.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include "LSystem.h"
#define MAX(a, b) ((a) < (b) ? (b) : (a))

Vector3D rot(0, -90, 0);
float camRotH, camRotV;
float camShH, camShV; // camera shift
int mouseOldX = 0;
int mouseOldY = 0;
LSystem lsystem;
unsigned systemList = 0; // display list to draw system
static int numIterations;	 // current # of iterations to draw system
float zoom = 1.0f;
int winWidth, winHeight;
bool saving = false;

/////////////////////////////////////////////////////////////////////////////////

void init()
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void reshape(int, int);
void saveImage();
void display()
{
    reshape(winWidth, winHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glRotatef(rot.x, 1, 0, 0);
    glRotatef(rot.y + camRotV, 0, 1, 0);
    glRotatef(rot.z + camRotH, 0, 0, 1);
    glScalef(zoom, zoom, zoom);

    if (systemList == 0) // no display list yet, build it )
    {
        systemList = glGenLists(1);

        glNewList(systemList, GL_COMPILE);
        lsystem.buildSystem(numIterations);
        lsystem.draw();
        glEndList();
    }

	Vector3D center = -lsystem.getBounds().getCenter();
	Vector3D size = lsystem.getBounds().getSize();

	float sizeMax = MAX(MAX(size.x, size.y), size.z);
	glScalef(4 / sizeMax, 4 / sizeMax, 4 / sizeMax);
	glTranslatef(center.x, center.y, center.z);

    glCallList(systemList);

    glPopMatrix();
    glutSwapBuffers();
    if (saving)
    {
        saving = false;
        saveImage();
    }
}

void reshape(int w, int h)
{
    winWidth = w;
    winHeight = h;
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    // factor all camera ops into projection matrix
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 60.0);
    Vector3D eye(5, 0, 0);
    eye += Vector3D(0, camShH, camShV);
    gluLookAt(eye.x, eye.y, eye.z, // eye
        0.0, camShH, camShV, // center
        0.0, 0.0, 1.0);	// up

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void motion(int x, int y)
{
    rot.y -= ((mouseOldY - y) * 180.0f) / 360.0f;
    rot.z -= ((mouseOldX - x) * 180.0f) / 360.0f;
    rot.x = 0;

    if (rot.z > 360)
        rot.z -= 360;

    if (rot.z < -360)
        rot.z += 360;

    if (rot.y > 360)
        rot.y -= 360;

    if (rot.y < -360)
        rot.y += 360;

    mouseOldX = x;
    mouseOldY = y;

    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouseOldX = x;
        mouseOldY = y;
    }
}

void mouseWheel(int wheel, int direction, int x, int y)
{
    wheel = 0;
    if (direction == -1 && zoom > 0.1f)
    {
        zoom -= 0.1f;

    }
    else if (direction == +1)
    {
        zoom += 0.1f;
    }

    glutPostRedisplay();
}

void processSpecialKeys(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_UP:
        camShV += 0.1f;
        break;
    case GLUT_KEY_DOWN:
        camShV -= 0.1f;
        break;
    case GLUT_KEY_LEFT:
        camShH -= 0.1f;
        break;
    case GLUT_KEY_RIGHT:
        camShH += 0.1f;
        break;
    }

    glutPostRedisplay();
}

void processKey(unsigned char key, int x, int y)
{
    if (key == 27 || key == 'q' || key == 'Q') //	quit requested
        exit(0);

    switch (key)
    {
    case '+': // increment # of iterations
        glDeleteLists(systemList, 1);
        systemList = 0;
        numIterations++;
        break;
    case '-': // decrement # of iterations
        glDeleteLists(systemList, 1);
        systemList = 0;
        if (numIterations > 0)
            numIterations--;
        break;
    case '=':
        camShH = 0.0f;
        camShV = 0.0f;
        break;
    case 'S':
    case 's':
        saving = true;
        break;
    case '8':
        camRotV -= 2.0f;
        break;
    case '2':
        camRotV += 2.0f;
        break;
    case '4':
        camRotH -= 2.0f;
        break;
    case '6':
        camRotH += 2.0f;
        break;
    }

    glutPostRedisplay();
}

void saveImage()
{

	time_t rawtime;
	struct tm* timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "screen_%Y.%m.%d_%Hh.%Mm.%Ss.png", timeinfo);
	puts(buffer);

    size_t width = winWidth;
    size_t height = winHeight;
    BYTE* pixels = new BYTE[3 * width * height];

    glReadPixels(0, 0, width, height, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels);

    // Convert to FreeImage format & save to file
    FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
    FreeImage_Save(FIF_PNG, image, buffer, 0);

    // Free resources
    FreeImage_Unload(image);
    delete[] pixels;
}

int main(int argc, char* argv[])
{
	ifstream in("input.bin", ios::binary);
	if (!in)
	{
		printf("Error of opening \"input.bin\"\n");
		return 1;
	}

	string initString;
	int rulesCount;
	map<char, string> rules;
	float width0, width1;
	unsigned char color0[4], color1[4];
	float angle, scaling;
	int temp;
	char* strTemp;

	in.read((char*)&rulesCount, sizeof(int)); // Rules count
	in.read((char*)&temp, sizeof(int)); // Init string length
	// Init string
	strTemp = new char[temp + 1];
	in.read((char*)strTemp, temp);
	strTemp[temp] = 0;
	initString = strTemp;
	delete[] strTemp;

	for (int i = 0; i < rulesCount; ++i)
	{
		// Rule letter
		char ruleName;
		in.read((char*)&ruleName, 1);
		// Rule string length
		in.read((char*)&temp, sizeof(int));
		// Relu definition
		strTemp = new char[temp + 1];
		in.read((char*)strTemp, temp);
		strTemp[temp] = 0;
		rules[ruleName] = strTemp;
		delete[] strTemp;
	}

	in.read((char*)&numIterations, sizeof(int)); // Start iteration
	in.read((char*)&color0, 4); // Color 0
	in.read((char*)&color1, 4); // Color 1
	in.read((char*)&width0, sizeof(float)); // Width 0
	in.read((char*)&width1, sizeof(float)); // Width 1
	in.read((char*)&angle, sizeof(float)); // Angle
	in.read((char*)&scaling, sizeof(float)); // Scaling
	in.close();

    // initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(500, 500);

    // create window
    glutCreateWindow("L-system demo");
    glutFullScreen();

    // register handlers
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(processSpecialKeys);
    glutKeyboardFunc(processKey);
    glutMouseFunc(mouse);
    glutMouseWheelFunc(mouseWheel);
    glutMotionFunc(motion);

    init();

    // initialize L-system
    lsystem.width0 = width0;
    lsystem.width1 = width1;
    // Color 0
    lsystem.color0[0] = color0[0] / 255.0f;
    lsystem.color0[1] = color0[1] / 255.0f;
    lsystem.color0[2] = color0[2] / 255.0f;
    lsystem.color0[3] = color0[3] / 255.0f;
    // Color 1
    lsystem.color1[0] = color1[0] / 255.0f;
    lsystem.color1[1] = color1[1] / 255.0f;
    lsystem.color1[2] = color1[2] / 255.0f;
    lsystem.color1[3] = color1[3] / 255.0f;
    lsystem.setInitialString(initString.c_str());
    for (auto r : rules)
        lsystem.addRule(r.first, r.second.c_str());
    lsystem.setAngle(angle);
    lsystem.setDistScale(scaling);

    glutMainLoop();

    return 0;
}
