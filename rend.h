#include "disp.h" /* include your own disp.h file (e.g. hw1)*/

/* Camera defaults */
#define	DEFAULT_FOV		35.0
#define	DEFAULT_IM_Z	(-10.0)  /* world coords for image plane origin */
#define	DEFAULT_IM_Y	(5.0)    /* default look-at point = 0,0,0 */
#define	DEFAULT_IM_X	(-10.0)

#define	DEFAULT_AMBIENT	{0.1, 0.1, 0.1}
#define	DEFAULT_DIFFUSE	{0.7, 0.6, 0.5}
#define	DEFAULT_SPECULAR	{0.2, 0.3, 0.4}
#define	DEFAULT_SPEC		32

#define	MATLEVELS	100		/* how many matrix pushes allowed */
#define	MAX_LIGHTS	10		/* how many lights allowed */

#define PI 3.14159265359
#ifndef GZRENDER
#define GZRENDER
typedef struct {			/* define a renderer */
  GzDisplay		*display;
  GzCamera		camera;
  short		    matlevel;	        /* top of stack - current xform */
  GzMatrix		Ximage[MATLEVELS];	/* stack of xforms (Xsm) */
  GzMatrix		Xnorm[MATLEVELS];	/* xforms for norms (Xim) */
  GzMatrix		Xsp;		        /* NDC to screen (pers-to-screen) */
  GzColor		flatcolor;          /* color state for flat shaded triangles */
  int			interp_mode;
  int			numlights;
  GzLight		lights[MAX_LIGHTS];
  GzLight		ambientlight;
  GzColor		Ka, Kd, Ks;
  float		    spec;		/* specular power */
  GzTexture		tex_fun;    /* tex_fun(float u, float v, GzColor color) */
}  GzRender;
#endif

typedef float vect4D[4];
typedef float TextureCoord[2];

// Function declaration
// HW2
int GzNewRender(GzRender **render, GzDisplay *display);
int GzFreeRender(GzRender *render);
int GzBeginRender(GzRender	*render);
int GzPutAttribute(GzRender	*render, int numAttributes, GzToken	*nameList, 
	GzPointer *valueList);
int GzPutTriangle(GzRender *render, int	numParts, GzToken *nameList,
	GzPointer *valueList);
short ctoi(float color);
void sortVerts(GzPointer listOfVerts, GzCoord &v1, GzCoord &v2, GzCoord &v3);
void computeEdge(const GzCoord &v1, const GzCoord &v2, GzCoord &E);
void computeBB(GzDisplay* display, const GzCoord &v1, const GzCoord &v2, const GzCoord &v3,
	int &ulx, int &uly, int &lrx, int &lry);
int floor(float x);
int computeLEE(const GzCoord &E, int y, int x);
GzDepth interpolateZ(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3, int y, int x);

// HW3
int GzPutCamera(GzRender *render, GzCamera *camera);
int GzPushMatrix(GzRender *render, GzMatrix	matrix);
int GzPopMatrix(GzRender *render);
void fillZeros(GzMatrix &A);
float degToRad(float theta);
void crossProd(const GzCoord &A, const GzCoord &B, GzCoord &C);
float norm(GzCoord A);
void matrixMult(GzMatrix A, GzMatrix B, GzMatrix &C);
void normalizeXn(GzMatrix &Xn);
void defineXiw(GzCamera* camera, GzMatrix &Xiw);
void Xform(GzMatrix &A, GzCoord &v1, GzCoord &v2);

// Object Translation
int GzRotXMat(float degree, GzMatrix mat);
int GzRotYMat(float degree, GzMatrix mat);
int GzRotZMat(float degree, GzMatrix mat);
int GzTrxMat(GzCoord translate, GzMatrix mat);
int GzScaleMat(GzCoord scale, GzMatrix mat);

// HW4
int signNum(float num);
void compColor(const GzRender *render, const GzCoord &N, GzColor &C);
float interpolateK(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3, int y, int x);

int textureLookup(const GzRender *render, const GzCoord &v1, const GzCoord &v2, const GzCoord &v3,
	const TextureCoord &txtr1PS, const TextureCoord &txtr2PS, const TextureCoord &txtr3PS, GzDepth Vzs,
	int y, int x, GzColor &Kt);

void interpolateCol(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3,
	const GzColor &c1, const GzColor &c2, const GzColor &c3, int y, int x, GzColor &col);

void interpolateNor(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3,
	const GzCoord &N1, const GzCoord &N2, const GzCoord &N3, int y, int x, GzCoord &N);
void txtrAffine2Persp(const TextureCoord &txtr, float Vzs, TextureCoord &txtrPS);
void txtrPersp2Affine(const TextureCoord &txtrPS, float Vzs, TextureCoord &txtr);
int flatShader(GzRender *render, int numParts, GzToken *nameList, GzPointer *valueList);
int gouraudShader(GzRender *render, int numParts, GzToken *nameList, GzPointer *valueList);
int phongShader(GzRender *render, int numParts, GzToken *nameList, GzPointer *valueList);

// HW5
int GzFreeTexture();
void compColorForTxtr(const GzRender *render, const GzCoord &N, GzColor &C);
void compColorForTxtr(const GzRender *render, const GzColor &Kt, const GzCoord &N, GzColor &C);

// Project
void addBump(GzColor color, float scale, GzCoord N);