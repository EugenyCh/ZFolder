#define _CRT_SECURE_NO_WARNINGS
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <FreeImage.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <string>
#include <sstream>
#include "Mandelbrot2D.cuh"
#define MAX(a, b) ((a) < (b) ? (b) : (a))

using namespace std;

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
int windowedWidth, windowedHeight;
bool fullscreen = false;
bool saving = false;

int fWindowSize;
int fFractalSize;
int fIterations;
int fMaxFractalSize;
int fGradientIndex;

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
        mandelbrot.fMaxFractalSize = fMaxFractalSize;
        mandelbrot.initColorSpectrum(fGradientIndex);
        mandelbrot.compute(fFractalSize, fFractalSize, fIterations, 1.0);
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
    //if (w != winWidth || h != winHeight)
    //    systemList = 0;

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
    rotY -= ((mouseOldY - y) * 180.0f) / 360.0f;
    rotZ -= ((mouseOldX - x) * 180.0f) / 360.0f;
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
    case 'F':
    case 'f':
        if (!fullscreen)
        {
            windowedWidth = winWidth;
            windowedHeight = winHeight;
            fullscreen = true;
            glutFullScreen();
        }
        break;
    case 'W':
    case 'w':
        if (fullscreen)
        {
            glutReshapeWindow(windowedWidth, windowedHeight);
            fullscreen = false;
        }
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
    strftime(buffer, 80, "screen_%Y.%m.%d_%Hh.%Mm.%Ss", timeinfo);
    stringstream ssname;
    ssname << buffer << "_s" << MIN(fFractalSize, fMaxFractalSize)
        << "_i" << fIterations
        << "_mand2d" << ".png";

    size_t width = winWidth;
    size_t height = winHeight;
    BYTE* pixels = new BYTE[3 * width * height];

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels);

    // Convert to FreeImage format & save to file
    FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);
    FreeImage_Save(FIF_PNG, image, ssname.str().c_str(), 0);

    // Free resources
    FreeImage_Unload(image);
    delete[] pixels;
    printf("Saved to %s\n", ssname.str().c_str());
}

int main(int argc, char* argv[])
{
    ifstream in("input.bin", ios::binary);
    if (!in)
    {
        printf("Error of opening \"input.bin\"\n");
        return 1;
    }

    in.read((char*)&fWindowSize, sizeof(int));
    in.read((char*)&fFractalSize, sizeof(int));
    in.read((char*)&fIterations, sizeof(int));
    in.read((char*)&fMaxFractalSize, sizeof(int));
    in.read((char*)&fGradientIndex, sizeof(int));
    in.close();

    // initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(fWindowSize, fWindowSize);

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
