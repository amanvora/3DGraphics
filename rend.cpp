/* CS580 Homework 5 */

#include	"stdafx.h"
#include	"stdio.h"
#include	"math.h"
#include	"Gz.h"
#include	"rend.h"

// returns A.B = A'B for 'dim'-D vectors
template<class T, int dim>
float dotProduct(const T &A, const T &B)
{
	float res = 0;
	for (int i = 0; i < dim; i++) {
		res += A[i] * B[i];
	}
	return res;
}

GzMatrix identityMatr =
{
	1.0,	0.0,	0.0,	0.0,
	0.0,	1.0,	0.0,	0.0,
	0.0,	0.0,	1.0,	0.0,
	0.0,	0.0,	0.0,	1.0
};

int GzRotXMat(float degree, GzMatrix mat)
{
// Create rotate matrix : rotate along x axis
// Pass back the matrix using mat value
	float theta = degToRad(degree);
	mat[0][0] = 1;			mat[0][1] = 0;			mat[0][2] = 0;				mat[0][3] = 0;
	mat[1][0] = 0;			mat[1][1] = cos(theta);	mat[1][2] = -sin(theta);	mat[1][3] = 0;
	mat[2][0] = 0;			mat[2][1] = sin(theta);	mat[2][2] = cos(theta);		mat[2][3] = 0;
	mat[3][0] = 0;			mat[3][1] = 0;			mat[3][2] = 0;				mat[3][3] = 1;
	return GZ_SUCCESS;
}


int GzRotYMat(float degree, GzMatrix mat)
{
// Create rotate matrix : rotate along y axis
// Pass back the matrix using mat value
	float theta = degToRad(degree);
	mat[0][0] = cos(theta);		mat[0][1] = 0;		mat[0][2] = sin(theta);		mat[0][3] = 0;
	mat[1][0] = 0;				mat[1][1] = 1;		mat[1][2] = 0;				mat[1][3] = 0;
	mat[2][0] = -sin(theta);	mat[2][1] = 0;		mat[2][2] = cos(theta);		mat[2][3] = 0;
	mat[3][0] = 0;				mat[3][1] = 0;		mat[3][2] = 0;				mat[3][3] = 1;
	return GZ_SUCCESS;
}


int GzRotZMat(float degree, GzMatrix mat)
{
// Create rotate matrix : rotate along z axis
// Pass back the matrix using mat value
	float theta = degToRad(degree);
	mat[0][0] = cos(theta);		mat[0][1] = -sin(theta);	mat[0][2] = 0;		mat[0][3] = 0;
	mat[1][0] = sin(theta);		mat[1][1] = cos(theta);		mat[1][2] = 0;		mat[1][3] = 0;
	mat[2][0] = 0;				mat[2][1] = 0;				mat[2][2] = 1;		mat[2][3] = 0;
	mat[3][0] = 0;				mat[3][1] = 0;				mat[3][2] = 0;		mat[3][3] = 1;
	return GZ_SUCCESS;
}


int GzTrxMat(GzCoord translate, GzMatrix mat)
{
// Create translation matrix
// Pass back the matrix using mat value
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if(i==j)
				mat[i][j] = 1;
			else
				mat[i][j] = 0;
		}
	}
	mat[X][3] = translate[X];
	mat[Y][3] = translate[Y];
	mat[Z][3] = translate[Z];
	return GZ_SUCCESS;
}


int GzScaleMat(GzCoord scale, GzMatrix mat)
{
// Create scaling matrix
// Pass back the matrix using mat value
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mat[i][j] = 0;
		}
	}
	mat[X][X] = scale[X];
	mat[Y][Y] = scale[Y];
	mat[Z][Z] = scale[Z];
	mat[3][3] = 1;
	return GZ_SUCCESS;
}


//----------------------------------------------------------
// Begin main functions

int GzNewRender(GzRender **render, GzDisplay	*display)
{
	// malloc a renderer struct 
	*render = new GzRender;

	// setup Xsp and anything only done once
	int xs, ys;
	xs = display->xres;
	ys = display->yres;
	fillZeros((*render)->Xsp);
	
	(*render)->Xsp[0][0] = (*render)->Xsp[0][3] = xs/2;
	(*render)->Xsp[1][1] = -ys/2;
	(*render)->Xsp[1][3] = ys/2;
	(*render)->Xsp[2][2] = INT_MAX;
	(*render)->Xsp[3][3] = 1;

	// Currently empty stack
	(*render)->matlevel = -1;

	// Setup I at TOS
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (i == j) {
				(*render)->Ximage[0][i][j] = 1;
				(*render)->Xnorm[0][i][j] = 1;
			}
			else {
				(*render)->Ximage[0][i][j] = 0;
				(*render)->Xnorm[0][i][j] = 0;
			}
		}
	}

	// Increment as one element is present now
	((*render)->matlevel)++;

	// -------------------- Camera defaults --------------------------- //
	// Default FOV
	(*render)->camera.FOV = DEFAULT_FOV;
	// Camera postion
	(*render)->camera.position[X] = DEFAULT_IM_X;
	(*render)->camera.position[Y] = DEFAULT_IM_Y;
	(*render)->camera.position[Z] = DEFAULT_IM_Z;

	// Camera lookat
	(*render)->camera.lookat[X] = 0;
	(*render)->camera.lookat[Y] = 0;
	(*render)->camera.lookat[Z] = 0;

	// Camera up vector
	(*render)->camera.worldup[X] = 0;
	(*render)->camera.worldup[Y] = 1;
	(*render)->camera.worldup[Z] = 0;
	// ---------------------------------------------------------------- //

	// save the pointer to display 
	(*render)->display = display;
	return GZ_SUCCESS;
}

int GzFreeRender(GzRender *render)
{
/* 
-free all renderer resources
*/	delete render;
	return GZ_SUCCESS;
}


int GzBeginRender(GzRender *render)
{
/*  
- setup for start of each frame - init frame buffer color,alpha,z
- compute Xiw and projection xform Xpi from camera definition 
- init Ximage - put Xsp at base of stack, push on Xpi and Xiw 
- now stack contains Xsw and app can push model Xforms when needed 
*/ 
	if (GzInitDisplay(render->display) == GZ_SUCCESS) {
		int status = 0;

		// Push Xsp to Image stack. Not pushed to Normal stack.
		status |= GzPushMatrix(render, render->Xsp); // TOS -> I*Xsp = Xsp

		// --------------------------- Define Xpi ------------------------- //
		fillZeros((render->camera).Xpi);
		(render->camera).Xpi[0][0] = (render->camera).Xpi[1][1] = (render->camera).Xpi[3][3] = 1;
		// Convert FOV to radians
		float theta = degToRad((render->camera).FOV);
		float oneOverD = tan(theta/2);
		(render->camera).Xpi[2][2] = (render->camera).Xpi[3][2] = oneOverD;
		// ---------------------------------------------------------------- //
		
		// Push Xpi to Image stack. Not pushed to Normal stack.
		status |= GzPushMatrix(render, (render->camera).Xpi); // TOS -> Xsp*Xpi
		
		// Define Xiw and then push it to stack
		defineXiw(&(render->camera), (render->camera).Xiw);
		status |= GzPushMatrix(render, (render->camera).Xiw); // TOS -> Xsp*Xpi*Xiw

		return status;
	}
	else
		return GZ_FAILURE;
}

int GzPutCamera(GzRender *render, GzCamera *camera)
{
/*
- overwrite renderer camera structure with new camera definition
*/
	render->camera = *camera;
	return GZ_SUCCESS;
}

int GzPushMatrix(GzRender *render, GzMatrix	matrix)
{
/*
- push a matrix onto the Ximage stack
- check for stack overflow
*/
	if (render->matlevel >= MATLEVELS)
		return GZ_FAILURE;
	else {
		render->matlevel++;
		matrixMult(render->Ximage[render->matlevel-1], matrix, render->Ximage[render->matlevel]);
		if (render->matlevel < 3) {
			// Do not push Xsp, Xpi into the norm stack. Push I instead.
			matrixMult(render->Xnorm[render->matlevel - 1], identityMatr, render->Xnorm[render->matlevel]);
		}
		else {
			matrixMult(render->Xnorm[render->matlevel - 1], matrix, render->Xnorm[render->matlevel]);
		}		
		return GZ_SUCCESS;
	}
}

int GzPopMatrix(GzRender *render)
{
/*
- pop a matrix off the Ximage stack
- check for stack underflow
*/
	if (render->matlevel < 0)
		return GZ_FAILURE;
	else
		return GZ_SUCCESS;
}


int GzPutAttribute(GzRender	*render, int numAttributes, GzToken	*nameList, 
	GzPointer	*valueList) /* void** valuelist */
{
/*
- set renderer attribute states (e.g.: GZ_RGB_COLOR default color)
- later set shaders, interpolaters, texture maps, and lights
*/
	for (int i = 0; i < numAttributes; i++) {
		// GZ_RGB_COLOR token
		if (nameList[i] == GZ_RGB_COLOR) {
			render->flatcolor[RED] = (*((GzColor*)valueList[i]))[RED];
			render->flatcolor[GREEN] = (*((GzColor*)valueList[i]))[GREEN];
			render->flatcolor[BLUE] = (*((GzColor*)valueList[i]))[BLUE];
		}

		// GZ_DIRECTIONAL_LIGHT
		if (nameList[i] == GZ_DIRECTIONAL_LIGHT) {
			(render->lights[i]).direction[X] = (*((GzLight*)valueList[i])).direction[X];
			(render->lights[i]).direction[Y] = (*((GzLight*)valueList[i])).direction[Y];
			(render->lights[i]).direction[Z] = (*((GzLight*)valueList[i])).direction[Z];

			(render->lights[i]).color[RED] = (*((GzLight*)valueList[i])).color[RED];
			(render->lights[i]).color[GREEN] = (*((GzLight*)valueList[i])).color[GREEN];
			(render->lights[i]).color[BLUE] = (*((GzLight*)valueList[i])).color[BLUE];

			render->numlights++;
		}

		// GZ_AMBIENT_LIGHT
		if (nameList[i] == GZ_AMBIENT_LIGHT) {
			(render->ambientlight).direction[X] = (*((GzLight*)valueList[i])).direction[X];
			(render->ambientlight).direction[Y] = (*((GzLight*)valueList[i])).direction[Y];
			(render->ambientlight).direction[Z] = (*((GzLight*)valueList[i])).direction[Z];

			(render->ambientlight).color[RED] = (*((GzLight*)valueList[i])).color[RED];
			(render->ambientlight).color[GREEN] = (*((GzLight*)valueList[i])).color[GREEN];
			(render->ambientlight).color[BLUE] = (*((GzLight*)valueList[i])).color[BLUE];
		}

		// GZ_DIFFUSE_COEFFICIENT
		if (nameList[i] == GZ_DIFFUSE_COEFFICIENT) {
			render->Kd[RED] = (*((GzColor*)valueList[i]))[RED];
			render->Kd[GREEN] = (*((GzColor*)valueList[i]))[GREEN];
			render->Kd[BLUE] = (*((GzColor*)valueList[i]))[BLUE];
		}

		// GZ_INTERPOLATE
		if (nameList[i] == GZ_INTERPOLATE) {
			render->interp_mode = *((int*)valueList[i]);
		}

		// GZ_AMBIENT_COEFFICIENT
		if (nameList[i] == GZ_AMBIENT_COEFFICIENT) {
			render->Ka[RED] = (*((GzColor*)valueList[i]))[RED];
			render->Ka[GREEN] = (*((GzColor*)valueList[i]))[GREEN];
			render->Ka[BLUE] = (*((GzColor*)valueList[i]))[BLUE];
		}

		// GZ_SPECULAR_COEFFICIENT
		if (nameList[i] == GZ_SPECULAR_COEFFICIENT) {
			render->Ks[RED] = (*((GzColor*)valueList[i]))[RED];
			render->Ks[GREEN] = (*((GzColor*)valueList[i]))[GREEN];
			render->Ks[BLUE] = (*((GzColor*)valueList[i]))[BLUE];
		}

		// GZ_DISTRIBUTION_COEFFICIENT
		if (nameList[i] == GZ_DISTRIBUTION_COEFFICIENT) {
			render->spec = *((float*)valueList[i]);
		}

		// GZ_TEXTURE_MAP
		if (nameList[i] == GZ_TEXTURE_MAP) {
			render->tex_fun = (GzTexture)valueList[i];
		}
	}
	return GZ_SUCCESS;
}

int GzPutTriangle(GzRender	*render, int numParts, GzToken *nameList, GzPointer	*valueList)
/* numParts : how many names and values */
{
	int status = 0;
	if (render->interp_mode == GZ_FLAT) {
		// Flat shading
		status = flatShader(render, numParts, nameList, valueList);
	}
	else if (render->interp_mode == GZ_COLOR) {
		// Gouraud shading
		status = gouraudShader(render, numParts, nameList, valueList);
	}
	else if (render->interp_mode == GZ_NORMALS) {
		// Phong shading
		status = phongShader(render, numParts, nameList, valueList);
	}

	return status;
}

/* NOT part of API - just for general assistance */
short	ctoi(float color)		/* convert float color to GzIntensity short */
{
  return(short)((int)(color * ((1 << 12) - 1)));
}

void sortVerts(GzPointer listOfVerts, GzCoord &v0, GzCoord &v1, GzCoord &v2) {
	v0[Y] = ((GzCoord*)listOfVerts)[0][Y];
	v1[Y] = ((GzCoord*)listOfVerts)[1][Y];
	v2[Y] = ((GzCoord*)listOfVerts)[2][Y];

	int minYIndex, midYIndex, maxYIndex;

	minYIndex = v0[Y] < v1[Y] ? 0 : 1;
	if (minYIndex == 0) {
		minYIndex = v0[Y] < v2[Y] ? 0 : 2;
	}
	else {
		minYIndex = v1[Y] < v2[Y] ? 1 : 2;
	}

	if (minYIndex == 0) {
		maxYIndex = v1[Y] > v2[Y] ? 1 : 2;
		midYIndex = v1[Y] < v2[Y] ? 1 : 2;
	}
	else if (minYIndex == 1) {
		maxYIndex = v0[Y] > v2[Y] ? 0 : 2;
		midYIndex = v0[Y] < v2[Y] ? 0 : 2;
	}
	else {
		maxYIndex = v0[Y] > v1[Y] ? 0 : 1;
		midYIndex = v0[Y] < v1[Y] ? 0 : 1;
	}

	v0[X] = ((((GzCoord*)listOfVerts))[minYIndex])[X];
	v0[Y] = ((((GzCoord*)listOfVerts))[minYIndex])[Y];
	v0[Z] = ((((GzCoord*)listOfVerts))[minYIndex])[Z];

	v1[X] = ((((GzCoord*)listOfVerts))[midYIndex])[X];
	v1[Y] = ((((GzCoord*)listOfVerts))[midYIndex])[Y];
	v1[Z] = ((((GzCoord*)listOfVerts))[midYIndex])[Z];

	v2[X] = ((((GzCoord*)listOfVerts))[maxYIndex])[X];
	v2[Y] = ((((GzCoord*)listOfVerts))[maxYIndex])[Y];
	v2[Z] = ((((GzCoord*)listOfVerts))[maxYIndex])[Z];
}

void computeEdge(const GzCoord &v1, const GzCoord &v2, GzCoord &E) {
	E[X] = v2[Y] - v1[Y]; // A = dY
	E[Y] = v1[X] - v2[X]; // B = -dX
	E[Z] = -E[Y] * v1[Y] - E[X] * v1[X]; // C = dX*Y - dY*X
}

void computeBB(GzDisplay* display, const GzCoord &v1, const GzCoord &v2, const GzCoord &v3,
	int &ulx, int &uly, int &lrx, int &lry) {
	float minX, minY, maxX, maxY;
	minX = v1[X] < v2[X] ? v1[X] : v2[X];
	minX = minX < v3[X] ? minX : v3[X];

	minY = v1[Y] < v2[Y] ? v1[Y] : v2[Y];
	minY = minY < v3[Y] ? minY : v3[Y];

	maxX = v1[X] > v2[X] ? v1[X] : v2[X];
	maxX = maxX > v3[X] ? maxX : v3[X];

	maxY = v1[Y] > v2[Y] ? v1[Y] : v2[Y];
	maxY = maxY > v3[Y] ? maxY : v3[Y];

	// Calculate bounding box corner position. Ensure they don't overshoot frame buffer limits
	ulx = floor(minX); if (ulx < 0) ulx = 0;
	uly = floor(minY); if (uly < 0) uly = 0;

	lrx = floor(maxX); if (lrx >(display->xres)) lrx = display->xres;
	lry = floor(maxY); if (lry >(display->yres)) lry = display->yres;
}

int floor(float x) {
	return ((int)x);
}

int computeLEE(const GzCoord &E, int y, int x) {
	float A = E[X];
	float B = E[Y];
	float C = E[Z];

	if ((A*x + B*y + C) >= 0)
		return 1;
	else
		return -1;
}

GzDepth interpolateZ(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3, int y, int x) {
	// Edge vectors
	GzCoord e, f;

	// 3D plane equation terms
	float A, B, C, D;

	// Compute the two edges vectors
	e[X] = v2[X] - v1[X];
	e[Y] = v2[Y] - v1[Y];
	e[Z] = v2[Z] - v1[Z];

	f[X] = v3[X] - v1[X];
	f[Y] = v3[Y] - v1[Y];
	f[Z] = v3[Z] - v1[Z];

	// Calculate cross product
	A = e[Y] * f[Z] - f[Y] * e[Z];
	B = -e[X] * f[Z] + f[X] * e[Z];
	C = e[X] * f[Y] - f[X] * e[Y];

	// Solve for D
	D = -(A*v1[X] + B*v1[Y] + C*v1[Z]); // v1 is a solution to the equation

										// Interpolate Z value
	GzDepth z = -(A*x + B*y + D) / C;

	return z;
}

void fillZeros(GzMatrix &A) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			A[i][j] = 0;
		}
	}
}

float degToRad(float theta) {
	return (theta*PI/180);
}

// C = AxB
void crossProd(const GzCoord &A, const GzCoord &B, GzCoord &C) {
	C[X] = A[Y]*B[Z] - B[Y]*A[Z];
	C[Y] = B[X]*A[Z] - A[X]*B[Z];
	C[Z] = A[X]*B[Y] - B[X]*A[Y];
}

// Magnitude of A = |A|
float norm(GzCoord A) {
	return (sqrt(A[X]*A[X] + A[Y]*A[Y] + A[Z]*A[Z]));
}

// A*B = C
void matrixMult(GzMatrix A, GzMatrix B, GzMatrix &C) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			C[i][j] = 0;
			for (int k = 0; k < 4; k++) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
}

// Used for making Xn a unitary rotation matrix and free from translations
void normalizeXn(GzMatrix &Xn) {

	// Make Xn a unitary rotation matrix
	GzCoord XnRow = { Xn[0][0], Xn[0][1], Xn[0][2] };
	float K = norm(XnRow);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Xn[i][j] /= K;
		}
	}

	// Remove all the translations from Xn
	Xn[0][3] = 0;
	Xn[1][3] = 0;
	Xn[2][3] = 0;
}

// Define Xiw
void defineXiw(GzCamera* camera, GzMatrix &Xiw) {

	// Obtain camera parameters
	GzCoord C = {(camera->position)[X], (camera->position)[Y], (camera->position)[Z]};
	GzCoord I = {(camera->lookat)[X], (camera->lookat)[Y], (camera->lookat)[Z]};
	GzCoord up = {(camera->worldup)[X], (camera->worldup)[Y], (camera->worldup)[Z] };

	GzCoord CI = { I[X] - C[X],
				   I[Y] - C[Y],
				   I[Z] - C[Z] };

	// Compute Z = CI/|CI|
	float modCI = norm(CI);
	GzCoord ZAxisCam = { CI[X]/ modCI, CI[Y]/ modCI, CI[Z]/ modCI };

	// Compute up.Z
	float upProjOnZ = dotProduct<GzCoord,3>(up,ZAxisCam);
	
	// Find up' = up - (up*Z)Z
	GzCoord upPrime = { up[X] - upProjOnZ*ZAxisCam[X],
						up[Y] - upProjOnZ*ZAxisCam[Y],
						up[Z] - upProjOnZ*ZAxisCam[Z] };
	float modUpPrime = norm(upPrime);
	GzCoord YAxisCam = { upPrime[X]/modUpPrime, upPrime[Y]/modUpPrime, upPrime[Z]/modUpPrime };

	// X = Y x Z
	GzCoord XAxisCam;
	crossProd(YAxisCam, ZAxisCam, XAxisCam);

	// Building Xiw
	Xiw[0][0] = XAxisCam[X];	Xiw[0][1] = XAxisCam[Y];	Xiw[0][2] = XAxisCam[Z];	Xiw[0][3] = -dotProduct<GzCoord, 3>(XAxisCam, C);;
	Xiw[1][0] = YAxisCam[X];	Xiw[1][1] = YAxisCam[Y];	Xiw[1][2] = YAxisCam[Z];	Xiw[1][3] = -dotProduct<GzCoord, 3>(YAxisCam, C);
	Xiw[2][0] = ZAxisCam[X];	Xiw[2][1] = ZAxisCam[Y];	Xiw[2][2] = ZAxisCam[Z];	Xiw[2][3] = -dotProduct<GzCoord, 3>(ZAxisCam, C);
	Xiw[3][0] = 0;				Xiw[3][1] = 0;				Xiw[3][2] = 0;				Xiw[3][3] = 1;
}

// Transforms v1 using A to v2
void Xform(GzMatrix &A, GzCoord &v1, GzCoord &v2) {
	vect4D V1 = {v1[X], v1[Y], v1[Z], 1};
	vect4D V2;
	for (int i = 0; i < 4; i++) {
		vect4D A_row = { A[i][0], A[i][1], A[i][2], A[i][3] };
		V2[i] = dotProduct<vect4D,4>(A_row, V1);
	}
	float w = V2[3];
	v2[X] = V2[X]/w;
	v2[Y] = V2[Y]/w;
	v2[Z] = V2[Z]/w;
};

int signNum(float num) {
	if (num > 0)return 1;
	else if (num == 0) return 0;
	else return -1;
}

// Calculate color using shading equation
void compColor(const GzRender *render, const GzCoord &N, GzColor &C) {

	GzColor specComp = { 0,0,0 };
	GzColor diffuseComp = { 0,0,0 };
	GzColor ambientComp;

	// Directional light
	const GzCoord* L;

	const GzColor* Ie;

	// Reflected ray R of L
	GzCoord R;

	// Direction of view of camera/eye
	GzCoord E = { 0,0,-1 };
	
	// Ambient component : Ka*la
	for (int c = 0; c < 3; c++) {
		ambientComp[c] = (render->Ka)[c] * ((render->ambientlight).color)[c];
	}

	// For each light calculate specular and diffuse component
	for (int l = 0; l < 3; l++) {
		
		L = &((render->lights[l]).direction);
		Ie = &((render->lights[l]).color);
		// Check whether light and eye are opposite or same side
		float NDotL = dotProduct<GzCoord, 3>(N, *L);
		float NDotE = dotProduct<GzCoord, 3>(N, E);
		if (signNum(NDotL) != signNum(NDotE)) {
			// Skip this light and continue with next
			continue;
		}
		
		//------------------- Specular component -----------------------//
		// Compute R
		for (int dir = 0; dir < 3; dir++) {
			R[dir] = 2*NDotL*N[dir] - (*L)[dir];
		}

		if((signNum(NDotL)==-1) && (signNum(NDotE)==-1)){
			// Both are negative
			// Flip N
			NDotL = -NDotL; // After fipping N, N.L has opposite sign
		}

		// Compute R.E and clamp it to [0,1]
		float RDotE = dotProduct<GzCoord, 3>(R, E);
		if (RDotE < 0) RDotE = 0;

		for (int c = 0; c < 3; c++) {
			// Add up specular component of this light
			specComp[c] += (render->Ks)[c] * ((*Ie)[c]) * pow(RDotE, render->spec);
		}

		//-------------------- Diffuse component -----------------------//
		for (int c = 0; c < 3; c++) {
			// Add up specular component of this light
			diffuseComp[c] += (render->Kd)[c] * ((*Ie)[c]) * NDotL;
		}
	}

	// Add up all the 3 components
	for (int c = 0; c < 3; c++) {
		C[c] = specComp[c] + diffuseComp[c] + ambientComp[c];
		if (C[c] > 1)C[c] = 1; // Avoid overflows greater than 1
		if (C[c] < 0)C[c] = 0; // Avoid overflows less than 0
	}
}

// Calculate color using shading equation but modified for texture
void compColorForTxtr(const GzRender *render, const GzCoord &N, GzColor &C) {

	GzColor specComp = { 0,0,0 };
	GzColor diffuseComp = { 0,0,0 };
	GzColor ambientComp;

	// Directional light
	const GzCoord* L;

	const GzColor* Ie;

	// Reflected ray R of L
	GzCoord R;

	// Direction of view of camera/eye
	GzCoord E = { 0,0,-1 };
	
	// Ambient component : la
	for (int c = 0; c < 3; c++) {
		ambientComp[c] = ((render->ambientlight).color)[c];
	}

	// For each light calculate specular and diffuse component
	for (int l = 0; l < 3; l++) {
		
		L = &((render->lights[l]).direction);
		Ie = &((render->lights[l]).color);
		// Check whether light and eye are opposite or same side
		float NDotL = dotProduct<GzCoord, 3>(N, *L);
		float NDotE = dotProduct<GzCoord, 3>(N, E);
		if (signNum(NDotL) != signNum(NDotE)) {
			// Skip this light and continue with next
			continue;
		}
		
		//------------------- Specular component -----------------------//
		// Compute R
		for (int dir = 0; dir < 3; dir++) {
			R[dir] = 2*NDotL*N[dir] - (*L)[dir];
		}

		if((signNum(NDotL)==-1) && (signNum(NDotE)==-1)){
			// Both are negative
			// Flip N
			NDotL = -NDotL; // After fipping N, N.L has opposite sign
		}

		// Compute R.E and clamp it to [0,1]
		float RDotE = dotProduct<GzCoord, 3>(R, E);
		if (RDotE < 0) RDotE = 0;

		for (int c = 0; c < 3; c++) {
			// Add up specular component of this light
			specComp[c] += ((*Ie)[c]) * pow(RDotE, render->spec);
		}

		//-------------------- Diffuse component -----------------------//
		for (int c = 0; c < 3; c++) {
			// Add up specular component of this light
			diffuseComp[c] += ((*Ie)[c]) * NDotL;
		}
	}

	// Add up all the 3 components
	for (int c = 0; c < 3; c++) {
		C[c] = specComp[c] + diffuseComp[c] + ambientComp[c];
		if (C[c] > 1)C[c] = 1; // Avoid overflows greater than 1
		if (C[c] < 0)C[c] = 0; // Avoid overflows less than 0
	}
}

// Calculate color using shading equation but modified for texture. This is used only for phong shading
void compColorForTxtr(const GzRender *render, const GzColor &Kt, const GzCoord &N, GzColor &C) {

	GzColor specComp = { 0,0,0 };
	GzColor diffuseComp = { 0,0,0 };
	GzColor ambientComp;

	// Directional light
	const GzCoord* L;

	const GzColor* Ie;

	// Reflected ray R of L
	GzCoord R;

	// Direction of view of camera/eye
	GzCoord E = { 0,0,-1 };
	
	// Ambient component : Ka*la. Ka = Kt
	for (int c = 0; c < 3; c++) {
		ambientComp[c] = Kt[c] * ((render->ambientlight).color)[c];
	}

	// For each light calculate specular and diffuse component
	for (int l = 0; l < 3; l++) {
		
		L = &((render->lights[l]).direction);
		Ie = &((render->lights[l]).color);
		// Check whether light and eye are opposite or same side
		float NDotL = dotProduct<GzCoord, 3>(N, *L);
		float NDotE = dotProduct<GzCoord, 3>(N, E);
		if (signNum(NDotL) != signNum(NDotE)) {
			// Skip this light and continue with next
			continue;
		}
		
		//------------------- Specular component -----------------------//
		// Compute R
		for (int dir = 0; dir < 3; dir++) {
			R[dir] = 2*NDotL*N[dir] - (*L)[dir];
		}

		if((signNum(NDotL)==-1) && (signNum(NDotE)==-1)){
			// Both are negative
			// Flip N
			NDotL = -NDotL; // After fipping N, N.L has opposite sign
		}

		// Compute R.E and clamp it to [0,1]
		float RDotE = dotProduct<GzCoord, 3>(R, E);
		if (RDotE < 0) RDotE = 0;

		for (int c = 0; c < 3; c++) {
			// Add up specular component of this light
			specComp[c] += (render->Ks)[c] * ((*Ie)[c]) * pow(RDotE, render->spec);
		}

		//-------------------- Diffuse component -----------------------//
		for (int c = 0; c < 3; c++) {
			// Add up specular component of this light. Kd = Kt
			diffuseComp[c] += Kt[c] * ((*Ie)[c]) * NDotL;
		}
	}

	// Add up all the 3 components
	for (int c = 0; c < 3; c++) {
		C[c] = specComp[c] + diffuseComp[c] + ambientComp[c];
		if (C[c] > 1)C[c] = 1; // Avoid overflows greater than 1
		if (C[c] < 0)C[c] = 0; // Avoid overflows less than 0
	}
}

// Flat shading
int flatShader(GzRender *render, int numParts, GzToken *nameList, GzPointer *valueList) {

	GzDisplay* display = render->display;
	GzCoord vert1, vert2, vert3;
	GzCoord norm1, norm2, norm3;
	TextureCoord txtr1, txtr2, txtr3;
	GzIntensity r, g, b;
	const int VERT_INDEX = 0;
	const int NORM_INDEX = 1;
	const int TEXTURE_INDEX = 2;
	
	//-------------------------------------------------------------------------------//
	//--------------------------- Process normals -----------------------------------//
	//-------------------------------------------------------------------------------//

	// Normalize top of stack
	normalizeXn(render->Xnorm[render->matlevel]);
	
	// Transform triangle vertices from model space to screen space
	GzMatrix* Xn;
	Xn = &(render->Xnorm[render->matlevel]);

	// Obtain normals in model space
	for (int dir = 0; dir < 3; dir++) {
		norm1[dir] = ((GzCoord*)valueList[NORM_INDEX])[0][dir];
		norm2[dir] = ((GzCoord*)valueList[NORM_INDEX])[1][dir];
		norm3[dir] = ((GzCoord*)valueList[NORM_INDEX])[2][dir];
	}

	// Apply transformation to normals. Model space to image space
	Xform(*Xn, norm1, norm1);
	Xform(*Xn, norm2, norm2);
	Xform(*Xn, norm3, norm3);

	// Obtain color using shading equation
	GzColor color;
	compColor(render, norm1, color);

	//-------------------------------------------------------------------------------//
	//--------------------------- Process vertices ----------------------------------//
	//-------------------------------------------------------------------------------//

	// Transform triangle vertices from model space to screen space
	GzMatrix* Xsm;
	Xsm = &(render->Ximage[render->matlevel]);

	GzCoord vertList[3];
	// Obtain vertices in model space
	for (int dir = 0; dir < 3; dir++) {
		vert1[dir] = ((GzCoord*)valueList[VERT_INDEX])[0][dir];
		vert2[dir] = ((GzCoord*)valueList[VERT_INDEX])[1][dir];
		vert3[dir] = ((GzCoord*)valueList[VERT_INDEX])[2][dir];
	}

	// Apply transformation. Model space to screen space
	Xform(*Xsm, vert1, vert1);
	Xform(*Xsm, vert2, vert2);
	Xform(*Xsm, vert3, vert3);

	// Ignore triangle if any vertex's Z is negative
	if ((vert1[Z] >= 0) && (vert2[Z] >= 0) && (vert3[Z] >= 0)) {
		//-----------------------------------------------------------//
		//----------------------- Rasterize -------------------------//
		//-----------------------------------------------------------//
		
		// Obtain u,v coords : Texture coords in affine space
		for (int dir = 0; dir < 2; dir++) {
			txtr1[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[0][dir];
			txtr2[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[1][dir];
			txtr3[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[2][dir];
		}

		// Obtain U,V coords : Texture coords in perspective space
		TextureCoord txtr1PS, txtr2PS, txtr3PS;
		txtrAffine2Persp(txtr1, vert1[Z], txtr1PS);
		txtrAffine2Persp(txtr2, vert2[Z], txtr2PS);
		txtrAffine2Persp(txtr3, vert3[Z], txtr3PS);

		// Concatenate all the vertices in screen space. To be used by sortVerts
		for (int dir = 0; dir < 3; dir++) {
			vertList[0][dir] = vert1[dir];
			vertList[1][dir] = vert2[dir];
			vertList[2][dir] = vert3[dir];
		}
		// Sort all the 3 vertices based on Y
		sortVerts(vertList, vert1, vert2, vert3);
		// Edges
		GzCoord E12, E23, E31;

		// Compute edge E12
		computeEdge(vert1, vert2, E12);

		// Compute edge E23
		computeEdge(vert2, vert3, E23);

		// Compute edge E31
		computeEdge(vert3, vert1, E31);

		// Bounding box corners
		int ulx, uly, lrx, lry;
		computeBB(display, vert1, vert2, vert3, ulx, uly, lrx, lry);

		for (int j = uly; j <= lry; j++) {
			for (int k = ulx; k <= lrx; k++) {

				// Compute LEE of (x,y) with E1
				int lEE1 = computeLEE(E12, j, k);

				// Compute LEE of (x,y) with E2
				int lEE2 = computeLEE(E23, j, k);

				// Compute LEE of (x,y) with E3
				int lEE3 = computeLEE(E31, j, k);

				// Check if all LEE were consitent
				if ((lEE1 + lEE2 + lEE3) == 3 || (lEE1 + lEE2 + lEE3) == -3) {
					GzDepth zPix = interpolateZ(vert1, vert2, vert3, j, k);
					GzDepth zBuff = display->fbuf[ARRAY(k, j)].z;
					if (zPix < zBuff) {
						GzColor Kt;
						textureLookup(render, vert1, vert2, vert3, txtr1PS, txtr2PS, txtr3PS, zPix, j, k, Kt);

						// Triangle color
						r = ctoi(Kt[RED]*color[RED]);
						g = ctoi(Kt[GREEN]*color[GREEN]);
						b = ctoi(Kt[BLUE]*color[BLUE]);
						GzPutDisplay(display, k, j, r, g, b, 1, zPix);
					}
				}
			}
		}
	}

	return GZ_SUCCESS;
}

float interpolateK(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3, int y, int x) {
	// Edge vectors
	GzCoord e, f;

	// 3D plane equation terms
	float A, B, C, D;

	// Compute the two edges vectors
	e[X] = v2[X] - v1[X];
	e[Y] = v2[Y] - v1[Y];
	e[Z] = v2[Z] - v1[Z];

	f[X] = v3[X] - v1[X];
	f[Y] = v3[Y] - v1[Y];
	f[Z] = v3[Z] - v1[Z];

	// Calculate cross product
	A = e[Y] * f[Z] - f[Y] * e[Z];
	B = -e[X] * f[Z] + f[X] * e[Z];
	C = e[X] * f[Y] - f[X] * e[Y];

	// Solve for D
	D = -(A*v1[X] + B*v1[Y] + C*v1[Z]); // v1 is a solution to the equation

	float k = -(A*x + B*y + D) / C; // Interpolate k value

	return k;
}

int textureLookup(const GzRender *render, const GzCoord &v1, const GzCoord &v2, const GzCoord &v3,
	const TextureCoord &txtr1PS, const TextureCoord &txtr2PS, const TextureCoord &txtr3PS, GzDepth Vzs, 
	int y, int x, GzColor &Kt) {

	TextureCoord txtrPS_xy, txtr_xy;
	GzCoord vert1U = { v1[X], v1[Y], txtr1PS[U] };
	GzCoord vert2U = { v2[X], v2[Y], txtr2PS[U] };
	GzCoord vert3U = { v3[X], v3[Y], txtr3PS[U] };
	txtrPS_xy[U] = interpolateK(vert1U, vert2U, vert3U, y, x);

	GzCoord vert1V = { v1[X], v1[Y], txtr1PS[V] };
	GzCoord vert2V = { v2[X], v2[Y], txtr2PS[V] };
	GzCoord vert3V = { v3[X], v3[Y], txtr3PS[V] };
	txtrPS_xy[V] = interpolateK(vert1V, vert2V, vert3V, y, x);

	txtrPersp2Affine(txtrPS_xy, Vzs, txtr_xy);

	int status = ((render->tex_fun))(txtr_xy[U], txtr_xy[V], Kt);
	return status;
}

void interpolateCol(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3, 
	const GzColor &c1, const GzColor &c2, const GzColor &c3, int y, int x, GzColor &col) {
	GzCoord vert1R = { v1[X], v1[Y], c1[RED] };
	GzCoord vert2R = { v2[X], v2[Y], c2[RED] };
	GzCoord vert3R = { v3[X], v3[Y], c3[RED] };
	col[RED] = interpolateK(vert1R, vert2R, vert3R, y, x);

	GzCoord vert1G = { v1[X], v1[Y], c1[GREEN] };
	GzCoord vert2G = { v2[X], v2[Y], c2[GREEN] };
	GzCoord vert3G = { v3[X], v3[Y], c3[GREEN] };
	col[GREEN] = interpolateK(vert1G, vert2G, vert3G, y, x);

	GzCoord vert1B = { v1[X], v1[Y], c1[BLUE] };
	GzCoord vert2B = { v2[X], v2[Y], c2[BLUE] };
	GzCoord vert3B = { v3[X], v3[Y], c3[BLUE] };
	col[BLUE] = interpolateK(vert1B, vert2B, vert3B, y, x);
}

void interpolateNor(const GzCoord &v1, const GzCoord &v2, const GzCoord &v3,
	const GzCoord &N1, const GzCoord &N2, const GzCoord &N3, int y, int x, GzCoord &N) {
	GzCoord vert1R = { v1[X], v1[Y], N1[X] };
	GzCoord vert2R = { v2[X], v2[Y], N2[X] };
	GzCoord vert3R = { v3[X], v3[Y], N3[X] };
	N[X] = interpolateK(vert1R, vert2R, vert3R, y, x);

	GzCoord vert1G = { v1[X], v1[Y], N1[Y] };
	GzCoord vert2G = { v2[X], v2[Y], N2[Y] };
	GzCoord vert3G = { v3[X], v3[Y], N3[Y] };
	N[Y] = interpolateK(vert1G, vert2G, vert3G, y, x);

	GzCoord vert1B = { v1[X], v1[Y], N1[Z] };
	GzCoord vert2B = { v2[X], v2[Y], N2[Z] };
	GzCoord vert3B = { v3[X], v3[Y], N3[Z] };
	N[Z] = interpolateK(vert1B, vert2B, vert3B, y, x);
}

void txtrAffine2Persp(const TextureCoord &txtr, float Vzs, TextureCoord &txtrPS) {
	float VzDash = Vzs / (INT_MAX - Vzs);
	txtrPS[U] = txtr[U] / (VzDash + 1);
	txtrPS[V] = txtr[V] / (VzDash + 1);
}

void txtrPersp2Affine(const TextureCoord &txtrPS, float Vzs, TextureCoord &txtr) {
	float VzDash = Vzs / (INT_MAX - Vzs);
	txtr[U] = txtrPS[U] * (VzDash + 1);
	txtr[V] = txtrPS[V] * (VzDash + 1);
}

// Gouraud shading
int gouraudShader(GzRender *render, int numParts, GzToken *nameList, GzPointer *valueList) {
	GzDisplay* display = render->display;
	GzCoord vert1, vert2, vert3;
	GzCoord norm1, norm2, norm3;
	TextureCoord txtr1, txtr2, txtr3;
	GzIntensity r, g, b;
	const int VERT_INDEX = 0;
	const int NORM_INDEX = 1;
	const int TEXTURE_INDEX = 2;

	//-------------------------------------------------------------------------------//
	//--------------------------- Process normals -----------------------------------//
	//-------------------------------------------------------------------------------//

	// Normalize top of stack
	normalizeXn(render->Xnorm[render->matlevel]);

	// Transform triangle vertices from model space to screen space
	GzMatrix* Xn;
	Xn = &(render->Xnorm[render->matlevel]);

	// Obtain normals in model space
	for (int dir = 0; dir < 3; dir++) {
		norm1[dir] = ((GzCoord*)valueList[NORM_INDEX])[0][dir];
		norm2[dir] = ((GzCoord*)valueList[NORM_INDEX])[1][dir];
		norm3[dir] = ((GzCoord*)valueList[NORM_INDEX])[2][dir];
	}

	// Apply transformation to normals. Model space to image space
	Xform(*Xn, norm1, norm1);
	Xform(*Xn, norm2, norm2);
	Xform(*Xn, norm3, norm3);

	// Obtain color using shading equation
	GzColor color1, color2, color3;

	if (render->tex_fun != 0) {
		compColorForTxtr(render, norm1, color1);
		compColorForTxtr(render, norm2, color2);
		compColorForTxtr(render, norm3, color3);
	}
	else {
		compColor(render, norm1, color1);
		compColor(render, norm2, color2);
		compColor(render, norm3, color3);
	}

	//-------------------------------------------------------------------------------//
	//--------------------------- Process vertices ----------------------------------//
	//-------------------------------------------------------------------------------//

	// Transform triangle vertices from model space to screen space
	GzMatrix* Xsm;
	Xsm = &(render->Ximage[render->matlevel]);

	GzCoord vertList[3];
	// Obtain vertices in model space
	for (int dir = 0; dir < 3; dir++) {
		vert1[dir] = ((GzCoord*)valueList[VERT_INDEX])[0][dir];
		vert2[dir] = ((GzCoord*)valueList[VERT_INDEX])[1][dir];
		vert3[dir] = ((GzCoord*)valueList[VERT_INDEX])[2][dir];
	}

	// Apply transformation. Model space to screen space
	Xform(*Xsm, vert1, vert1);
	Xform(*Xsm, vert2, vert2);
	Xform(*Xsm, vert3, vert3);

	// Ignore triangle if any vertex's Z is negative
	if ((vert1[Z] >= 0) && (vert2[Z] >= 0) && (vert3[Z] >= 0)) {
		//-----------------------------------------------------------//
		//----------------------- Rasterize -------------------------//
		//-----------------------------------------------------------//

		// Obtain u,v coords : Texture coords in affine space
		for (int dir = 0; dir < 2; dir++) {
			txtr1[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[0][dir];
			txtr2[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[1][dir];
			txtr3[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[2][dir];
		}

		// Obtain U,V coords : Texture coords in perspective space
		TextureCoord txtr1PS, txtr2PS, txtr3PS;
		txtrAffine2Persp(txtr1, vert1[Z], txtr1PS);
		txtrAffine2Persp(txtr2, vert2[Z], txtr2PS);
		txtrAffine2Persp(txtr3, vert3[Z], txtr3PS);

		// Concatenate all the vertices in screen space. To be used by sortVerts
		for (int dir = 0; dir < 3; dir++) {
			vertList[0][dir] = vert1[dir];
			vertList[1][dir] = vert2[dir];
			vertList[2][dir] = vert3[dir];
		}
		// Sort all the 3 vertices based on Y
		sortVerts(vertList, vert1, vert2, vert3);
		// Edges
		GzCoord E12, E23, E31;

		// Compute edge E12
		computeEdge(vert1, vert2, E12);

		// Compute edge E23
		computeEdge(vert2, vert3, E23);

		// Compute edge E31
		computeEdge(vert3, vert1, E31);

		// Bounding box corners
		int ulx, uly, lrx, lry;
		computeBB(display, vert1, vert2, vert3, ulx, uly, lrx, lry);

		// Restore the vertices order so that it is the same as that of the normals
		for (int dir = 0; dir < 3; dir++) {
			vert1[dir] = vertList[0][dir];
			vert2[dir] = vertList[1][dir];
			vert3[dir] = vertList[2][dir];
		}

		for (int j = uly; j <= lry; j++) {
			for (int k = ulx; k <= lrx; k++) {

				// Compute LEE of (x,y) with E1
				int lEE1 = computeLEE(E12, j, k);

				// Compute LEE of (x,y) with E2
				int lEE2 = computeLEE(E23, j, k);

				// Compute LEE of (x,y) with E3
				int lEE3 = computeLEE(E31, j, k);

				// Check if all LEE were consitent
				if ((lEE1 + lEE2 + lEE3) == 3 || (lEE1 + lEE2 + lEE3) == -3) {
					GzDepth zPix = interpolateZ(vert1, vert2, vert3, j, k);
					GzDepth zBuff = display->fbuf[ARRAY(k, j)].z;
					if (zPix < zBuff) {
						if (render->tex_fun != 0) {
							// Pixel interpolated color
							GzColor Kt, pixelColor;

							interpolateCol(vert1, vert2, vert3, color1, color2, color3, j, k, pixelColor);
							textureLookup(render, vert1, vert2, vert3, txtr1PS, txtr2PS, txtr3PS, zPix, j, k, Kt);

							for (int c = 0; c < 3; c++) {
								pixelColor[c] *= Kt[c];
							}
							r = ctoi(pixelColor[RED]);
							g = ctoi(pixelColor[GREEN]);
							b = ctoi(pixelColor[BLUE]);
						}
						else {
							// Pixel interpolated color
							GzColor pixelColor;
							interpolateCol(vert1, vert2, vert3, color1, color2, color3, j, k, pixelColor);
							r = ctoi(pixelColor[RED]);
							g = ctoi(pixelColor[GREEN]);
							b = ctoi(pixelColor[BLUE]);
						}
						GzPutDisplay(display, k, j, r, g, b, 1, zPix);
					}
				}
			}
		}
	}
	return GZ_SUCCESS;
}

void addBump(GzColor C, float scale, GzCoord N) {
	// Convert Kt to inverted grayscale 
	C[RED] = 1 - (0.21*C[RED] + 0.72*C[GREEN] + 0.07*C[BLUE]);
	C[GREEN] = C[RED];
	C[BLUE] = C[RED];

	// Add bump mapping and displace the interpolated normal
	for (int dir = 0; dir < 3; dir++) {
		N[dir] += scale*(C[dir] - 0.5); // Convert Kt from (0 to 1) to (-0.5 to 0.5)
	}

	// Re-Normalize
	float nNorm = norm(N);
	for (int dir = 0; dir < 3; dir++) {
		N[dir] /= nNorm;
	}
}

// Phong shading
int phongShader(GzRender *render, int numParts, GzToken *nameList, GzPointer *valueList) {
	GzDisplay* display = render->display;
	GzCoord vert1, vert2, vert3;
	GzCoord norm1, norm2, norm3;
	TextureCoord txtr1, txtr2, txtr3;

	GzIntensity r, g, b;
	const int VERT_INDEX = 0;
	const int NORM_INDEX = 1;
	const int TEXTURE_INDEX = 2;

	//-------------------------------------------------------------------------------//
	//--------------------------- Process normals -----------------------------------//
	//-------------------------------------------------------------------------------//

	// Normalize top of stack
	normalizeXn(render->Xnorm[render->matlevel]);

	// Transform triangle vertices from model space to screen space
	GzMatrix* Xn;
	Xn = &(render->Xnorm[render->matlevel]);

	// Obtain normals in model space
	for (int dir = 0; dir < 3; dir++) {
		norm1[dir] = ((GzCoord*)valueList[NORM_INDEX])[0][dir];
		norm2[dir] = ((GzCoord*)valueList[NORM_INDEX])[1][dir];
		norm3[dir] = ((GzCoord*)valueList[NORM_INDEX])[2][dir];
	}

	// Apply transformation to normals. Model space to image space
	Xform(*Xn, norm1, norm1);
	Xform(*Xn, norm2, norm2);
	Xform(*Xn, norm3, norm3);

	//-------------------------------------------------------------------------------//
	//--------------------------- Process vertices ----------------------------------//
	//-------------------------------------------------------------------------------//

	// Transform triangle vertices from model space to screen space
	GzMatrix* Xsm;
	Xsm = &(render->Ximage[render->matlevel]);

	GzCoord vertList[3];
	// Obtain vertices in model space
	for (int dir = 0; dir < 3; dir++) {
		vert1[dir] = ((GzCoord*)valueList[VERT_INDEX])[0][dir];
		vert2[dir] = ((GzCoord*)valueList[VERT_INDEX])[1][dir];
		vert3[dir] = ((GzCoord*)valueList[VERT_INDEX])[2][dir];
	}

	// Apply transformation. Model space to screen space
	Xform(*Xsm, vert1, vert1);
	Xform(*Xsm, vert2, vert2);
	Xform(*Xsm, vert3, vert3);

	// Ignore triangle if any vertex's Z is negative
	if ((vert1[Z] >= 0) && (vert2[Z] >= 0) && (vert3[Z] >= 0)) {
		//-----------------------------------------------------------//
		//----------------------- Rasterize -------------------------//
		//-----------------------------------------------------------//

		// Obtain u,v coords : Texture coords in affine space
		for (int dir = 0; dir < 2; dir++) {
			txtr1[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[0][dir];
			txtr2[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[1][dir];
			txtr3[dir] = ((TextureCoord*)valueList[TEXTURE_INDEX])[2][dir];
		}

		// Obtain U,V coords : Texture coords in perspective space
		TextureCoord txtr1PS, txtr2PS, txtr3PS;
		txtrAffine2Persp(txtr1, vert1[Z], txtr1PS);
		txtrAffine2Persp(txtr2, vert2[Z], txtr2PS);
		txtrAffine2Persp(txtr3, vert3[Z], txtr3PS);

		// Concatenate all the vertices in screen space. To be used by sortVerts
		for (int dir = 0; dir < 3; dir++) {
			vertList[0][dir] = vert1[dir];
			vertList[1][dir] = vert2[dir];
			vertList[2][dir] = vert3[dir];
		}
		// Sort all the 3 vertices based on Y
		sortVerts(vertList, vert1, vert2, vert3);
		// Edges
		GzCoord E12, E23, E31;

		// Compute edge E12
		computeEdge(vert1, vert2, E12);

		// Compute edge E23
		computeEdge(vert2, vert3, E23);

		// Compute edge E31
		computeEdge(vert3, vert1, E31);

		// Bounding box corners
		int ulx, uly, lrx, lry;
		computeBB(display, vert1, vert2, vert3, ulx, uly, lrx, lry);

		// Restore the vertices order so that it is the same as that of the normals
		for (int dir = 0; dir < 3; dir++) {
			vert1[dir] = vertList[0][dir];
			vert2[dir] = vertList[1][dir];
			vert3[dir] = vertList[2][dir];
		}

		for (int j = uly; j <= lry; j++) {
			for (int k = ulx; k <= lrx; k++) {

				// Compute LEE of (x,y) with E1
				int lEE1 = computeLEE(E12, j, k);

				// Compute LEE of (x,y) with E2
				int lEE2 = computeLEE(E23, j, k);

				// Compute LEE of (x,y) with E3
				int lEE3 = computeLEE(E31, j, k);

				// Check if all LEE were consitent
				if ((lEE1 + lEE2 + lEE3) == 3 || (lEE1 + lEE2 + lEE3) == -3) {
					GzDepth zPix = interpolateZ(vert1, vert2, vert3, j, k);
					GzDepth zBuff = display->fbuf[ARRAY(k, j)].z;
					if (zPix < zBuff) {
						if (render->tex_fun != 0) {
							GzColor Kt;
							// Pixel interpolated normal
							GzCoord pixelNormal;
							textureLookup(render, vert1, vert2, vert3, txtr1PS, txtr2PS, txtr3PS, zPix, j, k, Kt);
							interpolateNor(vert1, vert2, vert3, norm1, norm2, norm3, j, k, pixelNormal);

							// Normalize the interpolated normal at the pixel location
							float pnNorm = norm(pixelNormal);
							for (int dir = 0; dir < 3; dir++) {
								pixelNormal[dir] /= pnNorm;
							}
							
							// Obtain color using shading equation
							GzColor pixelColor;
							compColorForTxtr(render, Kt, pixelNormal, pixelColor);

							r = ctoi(pixelColor[RED]);
							g = ctoi(pixelColor[GREEN]);
							b = ctoi(pixelColor[BLUE]);
						}
						else {
							// Pixel interpolated normal
							GzCoord pixelNormal;
							interpolateNor(vert1, vert2, vert3, norm1, norm2, norm3, j, k, pixelNormal);

							// Normalize the interpolated normal at the pixel location
							float pnNorm = norm(pixelNormal);
							for (int dir = 0; dir < 3; dir++) {
								pixelNormal[dir] /= pnNorm;
							}

							// Obtain color using shading equation
							GzColor pixelColor;
							compColor(render, pixelNormal, pixelColor);

							r = ctoi(pixelColor[RED]);
							g = ctoi(pixelColor[GREEN]);
							b = ctoi(pixelColor[BLUE]);
						}
						GzPutDisplay(display, k, j, r, g, b, 1, zPix);
					}
				}
			}
		}
	}
	return GZ_SUCCESS;
}