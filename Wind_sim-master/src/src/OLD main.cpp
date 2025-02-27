/********************************/
/* Ruled surface                */
/* (C) Bedrich Benes 2021       */
/* bbenes@purdue.edu     */
/********************************/

#include <stdio.h>
#include <iostream>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <string>
#include <vector>			//Standard template library
#include <array>
#include <GL/glew.h>
#include <GL/freeglut.h>

//in house created libraries
#include "math/vect3d.h"    //for vector manipulation
#include "math/triangle.h"  //triangles
#include "helper.h"         
#include "objGen.h"         //to save OBJ file format for 3D printing
#include "trackball.h"

#pragma warning(disable : 4996)
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "freeglut.lib")

using namespace std;

TrackBallC trackball;
bool mouseLeft, mouseMid, mouseRight;


vector <TriangleC> tri;   //all the triangles will be stored here


GLuint points=0; //number of points to display the object
int steps=12;//# of subdivisions
bool needRedisplay=false;
GLfloat  sign=+1;
const GLfloat defaultIncrement=0.03;
GLfloat  angleIncrement=defaultIncrement;


//the main window size
GLint wWindow=800;
GLint hWindow=600;

inline void AddVertex(vector <GLfloat> *a, const GLfloat *A)
{
	a->push_back(A[0]);a->push_back(A[1]);a->push_back(A[2]);
}


Vect3d P(GLfloat t)
{
	return Vect3d(0.3*cos(2 * M_PI*t+M_PI/2), 0, 0.6*sin(2 * M_PI*t+M_PI/2));
}

inline Vect3d Q(GLfloat t)
{
	return Vect3d(0.6*cos(2 * M_PI*t), 1, 0.6*sin(2*M_PI*t));
}

inline Vect3d S(GLfloat u, GLfloat t)
{
	return Vect3d(u*P(t)+(1-u)*Q(t));
}

void CreateRuled(vector <GLfloat> *vv, int n)
{
	GLfloat step=1.f/n;
	for (int i=0;i<n;i++)
	{
		for (int j=0;j<n;j++)
		{
			//lower triangle
			AddVertex(vv, S(i*step, j*step));
			AddVertex(vv, S((i + 1)*step, j*step));
			AddVertex(vv, S((i + 1)*step, (j + 1)*step));
			//upper triangle
			AddVertex(vv, S(i*step, j*step));
			AddVertex(vv, S((i + 1)*step, (j + 1)*step));
			AddVertex(vv, S(i*step, (j + 1)*step));
		}
	}
}


void InitArray(int n)
{
	GLuint vaID;

	vector<GLfloat> v;
	CreateRuled(&v,n); 
    //now get it ready for saving as OBJ
	tri.clear();
	for (unsigned int i = 0; i < v.size(); i += 9) { //stride 3 - 3 vertices per triangle
		TriangleC tmp;
		Vect3d a,b,c;
		a.Set(v[i], v[i + 1], v[i + 2]);
		b.Set(v[i+3], v[i + 4], v[i + 5]);
		c.Set(v[i+6], v[i + 7], v[i + 8]);
		tmp.Set(a,b,c); //store them for 3D printing
		tri.push_back(tmp);
	}

	glGenVertexArrays(1,&vaID);
	glBindVertexArray(vaID);

	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	points=v.size();
	glBufferData(GL_ARRAY_BUFFER, points*sizeof(GLfloat), &v[0], GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(0);
	v.clear(); //no need for the data, it is on the GPU now
}




//the main rendering function
void RenderObjects()
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	glColor3f(0,0,1);
	glPointSize(2);
	glLineWidth(1);
	trackball.Set3DViewCamera();
	CoordSyst();
	glDrawArrays(GL_TRIANGLES, 0, points/3);
}
	

void Idle(void)
{
  glClearColor(1.9,1.9,1.9,1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  GLMessage("Ruled Surface");
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(40,(GLfloat)wWindow/(GLfloat)hWindow,0.01,100); //set the camera
  glMatrixMode(GL_MODELVIEW); //set the scene
  glLoadIdentity();
  gluLookAt(0,5,5,0,0,0,0,1,0);
  static GLfloat angle=0;
  angle+=angleIncrement;
  if (angle>=360.f) angle=0.f;
  glRotatef(sign*angle,0,1,0);
  RenderObjects();
  glutSwapBuffers();  
}

void Kbd(unsigned char a, int x, int y)//keyboard callback
{
	switch(a)
	{
 	  case 27 : exit(0);break;
	  case 32 : {
		  if (angleIncrement==0) angleIncrement=defaultIncrement;
						    else angleIncrement=0;
		  break;
		}
	  case 's': {sign=-sign;break;}
	  case '-': {
				  steps--; 
				  if (steps<1) steps=1;
				  InitArray(steps);
				  break;
				}
	  case '+': {
				  steps++; 
				  InitArray(steps);
				  break;
				}
	  case 'w': {
		  SaveOBJ(&tri, "geometry.obj"); break; 
				}
	}
	cout << "[steps]=[" << steps << "]" << endl;
	glutPostRedisplay();
}


void Mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseLeft = true;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		trackball.Set(false, x, y);
		mouseLeft = false;
	}
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseMid = true;
	}
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_UP)
	{
		trackball.Set(true, x, y);
		mouseMid = false;
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseRight = true;
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
	{
		trackball.Set(true, x, y);
		mouseRight = false;
	}
}

void MouseMotion(int x, int y) {
	if (mouseLeft)  trackball.Rotate(x, y);
	if (mouseMid)   trackball.Translate(x, y);
	if (mouseRight) trackball.Zoom(x, y);
	glutPostRedisplay();
}


int main(int argc, char **argv)
{ 
  glutInitDisplayString("stencil>=2 rgb double depth samples");
  glutInit(&argc, argv);
  glutInitWindowSize(wWindow,hWindow);
  glutInitWindowPosition(500,100);
  glutCreateWindow("Ruled Surface");
  GLenum err = glewInit();
  if (GLEW_OK != err){
   fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }
  glutDisplayFunc(Display);
  glutIdleFunc(Idle);
  glutReshapeFunc(Reshape);
  glutKeyboardFunc(Kbd); //+ and -
  glutMouseFunc(NULL);
  glutSpecialUpFunc(NULL); 
  glutSpecialFunc(NULL);
  glutMouseFunc(Mouse);
  glutMotionFunc(MouseMotion);
  InitArray(steps);
  glutMainLoop();
  return 0;        
}
