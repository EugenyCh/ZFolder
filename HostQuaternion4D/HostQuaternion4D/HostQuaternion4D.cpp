#define _CRT_SECURE_NO_WARNINGS
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <stdio.h>
#include "QFractal.h"
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

float rotX = 0;
float rotY = 90;
float rotZ = 90;
float camRotH, camRotV;
float camShH, camShV; // camera shift
int mouseOldX = 0;
int mouseOldY = 0;
static QFractal qfractal(-0.65, -0.5, 0.0, 0.0, QFractal::C, 15);
unsigned systemList = 0; // display list to draw system
float zoom = 1.0f;
int winWidth, winHeight;

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
void display()
{
    reshape(winWidth, winHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glRotatef(rotX, 1, 0, 0);
    glRotatef(rotY + camRotV, 0, 1, 0);
    glRotatef(rotZ + camRotH, 0, 0, 1);
    glScalef(zoom, zoom, zoom);

    if (systemList == 0)
    {
        systemList = glGenLists(1);

        glNewList(systemList, GL_COMPILE);
        qfractal.compute(winWidth, winHeight);
        qfractal.initColorSpectrum();
        qfractal.draw(winWidth, winHeight);
        glEndList();
    }

    int winSide = MIN(winWidth, winHeight);
    glScalef(3.0f / winSide, 3.0f / winSide, 3.0f / winSide);
    glCallList(systemList);

    glPopMatrix();
    glutSwapBuffers();
}

void reshape(int w, int h)
{
    if (w != winWidth || h != winHeight)
        systemList = 0;

    winWidth = w;
    winHeight = h;
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    // factor all camera ops into projection matrix
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 60.0);
    float eyeX = 5;
    float eyeY = camShH;
    float eyeZ = camShV;
    gluLookAt(eyeX, eyeY, eyeZ, // eye
        0.0, camShH, camShV, // center
        0.0, 0.0, 1.0);	// up

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void motion(int x, int y)
{
    rotY -= ((mouseOldY - y) * 180.0f) / 200.0f;
    rotZ -= ((mouseOldX - x) * 180.0f) / 200.0f;
    rotX = 0;

    if (rotZ > 360)
        rotZ -= 360;

    if (rotZ < -360)
        rotZ += 360;

    if (rotY > 360)
        rotY -= 360;

    if (rotY < -360)
        rotY += 360;

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
    case '=':
        camShH = 0.0f;
        camShV = 0.0f;
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

int main(int argc, char* argv[])
{
    // initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(400, 400);

    // create window
    glutCreateWindow("Quaternion fractal demo");

    // register handlers
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(processSpecialKeys);
    glutKeyboardFunc(processKey);
    glutMouseFunc(mouse);
    glutMouseWheelFunc(mouseWheel);
    glutMotionFunc(motion);

    init();

    glutMainLoop();

    return 0;
}
