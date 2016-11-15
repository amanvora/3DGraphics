/* Texture functions for cs580 GzLib	*/
#include    "stdafx.h" 
#include	"stdio.h"
#include	"Gz.h"

GzColor	*image=NULL;
int xs, ys;
int reset = 1;
int W = 500, H = 500;
int N = 200;

typedef float complex[2];
/* Image texture function */
int tex_fun(float u, float v, GzColor color)
{
  unsigned char		pixel[3];
  unsigned char     dummy;
  char  		foo[8];
  int   		i, j;
  FILE			*fd;

  if (reset) {          /* open and load texture file */
    fd = fopen ("texture", "rb");
    if (fd == NULL) {
      fprintf (stderr, "texture file not found\n");
      exit(-1);
    }
    fscanf (fd, "%s %d %d %c", foo, &xs, &ys, &dummy);
    image = (GzColor*)malloc(sizeof(GzColor)*(xs+1)*(ys+1));
    if (image == NULL) {
      fprintf (stderr, "malloc for texture image failed\n");
      exit(-1);
    }

    for (i = 0; i < xs*ys; i++) {	/* create array of GzColor values */
      fread(pixel, sizeof(pixel), 1, fd);
      image[i][RED] = (float)((int)pixel[RED]) * (1.0 / 255.0);
      image[i][GREEN] = (float)((int)pixel[GREEN]) * (1.0 / 255.0);
      image[i][BLUE] = (float)((int)pixel[BLUE]) * (1.0 / 255.0);
      }

    reset = 0;          /* init is done */
	fclose(fd);
  }

/* bounds-test u,v to make sure nothing will overflow image array bounds */
  if (u < 0)u = 0;	if (u > 1)u = 1;
  if (v < 0)v = 0;	if (v > 1)v = 1;

/* determine texture cell corner values and perform bilinear interpolation */
  //----------------------- Scaling -------------------------//
  float scaled_u, scaled_v;
  scaled_u = u*(xs-1);
  scaled_v = v*(ys-1);

  //--------------------------------------------------------//
  //---------------- Bilinear Interpolation ----------------//
  float s, t;
  s = scaled_u - int(scaled_u);
  t = scaled_v - int(scaled_v);

  // Obtain all the four corners
  GzColor A, B, C, D;

  // Top left corner
  for (int c = 0; c < 3; c++) {
	  int index = int(scaled_u) + int(scaled_v)*xs;
	  A[c] = image[index][c];
  }
  
  // Top right corner
  for (int c = 0; c < 3; c++) {
	  int index = (int(scaled_u)+1) + int(scaled_v)*xs;
	  B[c] = image[index][c];
  }

  // Bottom left corner
  for (int c = 0; c < 3; c++) {
	  int index = int(scaled_u) + (int(scaled_v)+1)*xs;
	  C[c] = image[index][c];
  }

  // Bottom right corner
  for (int c = 0; c < 3; c++) {
	  int index = (int(scaled_u)+1) + (int(scaled_v)+1)*xs;
	  D[c] = image[index][c];
  }
  //---------------------------------------------------------//

/* set color to interpolated GzColor value and return */
  for (int c = 0; c < 3; c++) {
	  color[c] = s*t*C[c] + (1-s)*t*D[c] + s*(1-t)*B[c] + (1-s)*(1-t)*A[c];
  }
  return GZ_SUCCESS;
}

// Used in procedural texture
void F(const complex X_J, const complex C, complex &newX_J) {
	newX_J[U] = X_J[U]*X_J[U] - X_J[V]*X_J[V]; // Real part
	newX_J[V] = 2*X_J[U]*X_J[V]; // Imaginary part

	// Add C
	newX_J[U] += C[U];
	newX_J[V] += C[V];
}

/* Procedural texture function */
int ptex_fun(float u, float v, GzColor color)
{
	complex X_Julia = {2*u-1, 2*v-1};
	static complex C = {-0.8, 0.156};
	static GzColor colorLUT[11] = {
		{ 0.0462,    0.1869,    0.4984 },
		{ 0.0971,    0.4898,    0.9597 },
		{ 0.8235,    0.4456,    0.3404 },
		{ 0.6948,    0.6463,    0.5853 },
		{ 0.3171,    0.7094,    0.2238 },
		{ 0.9502,    0.7547,    0.7513 },
		{ 0.0344,    0.2760,    0.2551 },
		{ 0.4387,    0.6797,    0.5060 },
		{ 0.3816,    0.6551,    0.6991 },
		{ 0.7655,    0.1626,    0.8909 },
		{ 0.7952,    0.1190,    0.9593 }
	};
	int i = 0;
	float length = sqrt(X_Julia[U] * X_Julia[U] + X_Julia[V] * X_Julia[V]);
	complex newX_Julia = { X_Julia[U], X_Julia[V] };
	while (i < N && length < 2) {
		X_Julia[U] = newX_Julia[U];
		X_Julia[V] = newX_Julia[V];
		F(X_Julia, C, newX_Julia);
		length = sqrt(newX_Julia[U] * newX_Julia[U] + newX_Julia[V] * newX_Julia[V]);
		i++;
	}
	float z = float(i) / N;
	int prevIndex, nextIndex;
	prevIndex = int(10*z);
	nextIndex = prevIndex+1;
	float a = nextIndex - 10*z;
	float b = 10*z - prevIndex;

	for (int c = 0; c < 3; c++) {
		color[c] = a*colorLUT[prevIndex][c] + b*colorLUT[nextIndex][c];
	}

	return GZ_SUCCESS;
}

/* Free texture memory */
int GzFreeTexture()
{
	if(image!=NULL)
		free(image);
	return GZ_SUCCESS;
}

