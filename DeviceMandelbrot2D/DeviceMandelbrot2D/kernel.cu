#define _CRT_SECURE_NO_WARNINGS
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <FreeImage.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include "Mandelbrot2D.cuh"
#define MAX(a, b) ((a) < (b) ? (b) : (a))

float rotX = 0;
float rotY = 90;
float rotZ = 90;
float camRotH, camRotV;
float camShH, camShV; // camera shift
int mouseOldX = 0;
int mouseOldY = 0;
static Mandelbrot2D mandelbrot;
unsigned systemList = 0; // display list to draw system
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

    glRotatef(rotX, 1, 0, 0);
    glRotatef(rotY + camRotV, 0, 1, 0);
    glRotatef(rotZ + camRotH, 0, 0, 1);
    glScalef(zoom, zoom, zoom);

    if (systemList == 0)
    {
        systemList = glGenLists(1);

        glNewList(systemList, GL_COMPILE);
        mandelbrot.initColorSpectrum();
        mandelbrot.compute(winWidth, winHeight, 200, 2.0);
        mandelbrot.draw();
        glEndList();
    }

    float scale = 5.0 / MIN(winWidth, winHeight);
    glScalef(scale, scale, scale);
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
    // initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(960, 640);

    // create window
    glutCreateWindow("Mandelbrot2D demo");

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
