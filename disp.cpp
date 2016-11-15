/*   CS580 HW1 display functions to be completed   */

#include   "stdafx.h"  
#include	"Gz.h"
#include	"disp.h"

int GzNewFrameBuffer(char** framebuffer, int width, int height)
{
/* HW1.1 create a framebuffer for MS Windows display:
 -- allocate memory for framebuffer : 3 bytes(b, g, r) x width x height
 -- pass back pointer 
 */
	// dynamically allocate memory
	*framebuffer = new char[3*width*height];
	return GZ_SUCCESS;
}

int GzNewDisplay(GzDisplay	**display, int xRes, int yRes)
{
/* HW1.2 create a display:
  -- allocate memory for indicated resolution
  -- pass back pointer to GzDisplay object in display
*/
	// Display is a pointer to a single GzDisplay struct
	*display = new GzDisplay;
	// fbuf is allocated memory for GzPixels
	(*display)->fbuf = new GzPixel[xRes*yRes];

	// Define display xres and yres
	(*display)->xres = xRes;
	(*display)->yres = yRes;
	return GZ_SUCCESS;
}


int GzFreeDisplay(GzDisplay	*display)
{
/* HW1.3 clean up, free memory */
	delete[] display->fbuf; // First delete memory pointed to by fbuf
	delete display;
	return GZ_SUCCESS;
}


int GzGetDisplayParams(GzDisplay *display, int *xRes, int *yRes)
{
/* HW1.4 pass back values for a display */
	*xRes = display->xres;
	*yRes = display->yres;
	return GZ_SUCCESS;
}


int GzInitDisplay(GzDisplay	*display)
{
/* HW1.5 set everything to some default values - start a new frame */
	// Default xres and yres
	display->xres = 256;
	display->yres = 256;
	// Assign default values to each GzPixel
	for (int i = 0; i < display->yres; i++) {
		for (int j = 0; j < display->xres; j++) {
			display->fbuf[ARRAY(j, i)].red = 2047;
			display->fbuf[ARRAY(j, i)].green = 2047;
			display->fbuf[ARRAY(j, i)].blue = 2047;
			display->fbuf[ARRAY(j, i)].alpha = 1;
			display->fbuf[ARRAY(j, i)].z = INT_MAX; // Default value of z
		}
	}
	return GZ_SUCCESS;
}


int GzPutDisplay(GzDisplay *display, int i, int j, GzIntensity r, GzIntensity g, GzIntensity b, GzIntensity a, GzDepth z)
{
/* HW1.6 write pixel values into the display */

	// Discard if coordinate is out of bounds
	if ((i >= 0) && (i < display->xres) &&
		(j >= 0) && (j < display->yres)
		) {
		// Clamp RGB values to range [0-4095]
		if (r > 4095) r = 4095; if (r < 0) r = 0;
		if (g > 4095) g = 4095; if (g < 0) g = 0;
		if (b > 4095) b = 4095; if (b < 0) b = 0;

		display->fbuf[ARRAY(i, j)].red = r;
		display->fbuf[ARRAY(i, j)].green = g;
		display->fbuf[ARRAY(i, j)].blue = b;
		display->fbuf[ARRAY(i, j)].alpha = a;
		display->fbuf[ARRAY(i, j)].z = z;
	}
	return GZ_SUCCESS;
}


int GzGetDisplay(GzDisplay *display, int i, int j, GzIntensity *r, GzIntensity *g, GzIntensity *b, GzIntensity *a, GzDepth *z)
{
/* HW1.7 pass back a pixel value to the display */
	// Return the values of GzPixel at (i, j)
	*r = display->fbuf[ARRAY(i, j)].red;
	*g = display->fbuf[ARRAY(i, j)].green;
	*b = display->fbuf[ARRAY(i, j)].blue;
	*a = display->fbuf[ARRAY(i, j)].alpha;
	*z = display->fbuf[ARRAY(i, j)].z;
	return GZ_SUCCESS;
}


int GzFlushDisplay2File(FILE* outfile, GzDisplay *display)
{

/* HW1.8 write pixels to ppm file -- "P6 %d %d 255\r" */
	
	int headerSize = sizeof("P6 256 256 255\r");
	int bytesToWrite = (headerSize-1) + 3*display->xres*display->yres;
	char *dataToWrite;
	
	GzNewFrameBuffer(&dataToWrite, bytesToWrite, 1);
	// Write bufferheader
	sprintf(dataToWrite, "P6 %d %d 255\r", display->xres, display->yres);

	char* framePtr = dataToWrite + (headerSize-1);
	
	// Pixels values to be written in RGBRGBRGB.... format
	for (int i = 0; i < display->yres; i++) {
		for (int j = 0; j < display->xres; j++) {

			GzIntensity rShort, gShort, bShort;
			// Right shift by 4 bits
			rShort = (display->fbuf[ARRAY(i, j)].red) >> 4;
			gShort = (display->fbuf[ARRAY(i, j)].green) >> 4;
			bShort = (display->fbuf[ARRAY(i, j)].blue) >> 4;
			
			// Consider only the lower byte
			framePtr[3 * ARRAY(i, j)] = (char)(rShort & 0xFF);
			framePtr[(3 * ARRAY(i, j)) + 1] = (char)(gShort & 0xFF);
			framePtr[(3 * ARRAY(i, j)) + 2] = (char)(bShort & 0xFF);
		}
	}

	// Write the data(i.e, bufferheader+pixels) to the output file
	fwrite(dataToWrite, sizeof(char), bytesToWrite, outfile);
	return GZ_SUCCESS;
}

int GzFlushDisplay2FrameBuffer(char* framebuffer, GzDisplay *display)
{

/* HW1.9 write pixels to framebuffer: 
	- put the pixels into the frame buffer
	- CAUTION: when storing the pixels into the frame buffer, the order is blue, green, and red 
	- NOT red, green, and blue !!!
*/
	// Pixels values to be written in BGR format to display
	for (int i = 0; i < display->yres; i++) {
		for (int j = 0; j < display->xres; j++) {

			GzIntensity rShort, gShort, bShort;
			// Right shift by 4 bits
			bShort = (display->fbuf[ARRAY(i, j)].blue) >> 4;
			gShort = (display->fbuf[ARRAY(i, j)].green) >> 4;
			rShort = (display->fbuf[ARRAY(i, j)].red) >> 4;

			// Consider only the lower byte
			framebuffer[3*ARRAY(i, j)] = (char)(bShort & 0xFF);
			framebuffer[(3*ARRAY(i, j))+1] = (char)(gShort & 0xFF);
			framebuffer[(3*ARRAY(i, j))+2] = (char)(rShort & 0xFF);
		}
	}
	return GZ_SUCCESS;
}