#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5
#define CONTRAST_THRESHOLD 10

/**
 * Detrimes if a pixel is maximum in
 *
 * @param expression
 *            the string to check
 * @return result of the induced calculation
 */
bool isMaximum(vector<vector<Mat> >& dog_pyr, int octave, int interval, int r,
		int c) {
	float intensity = dog_pyr[octave][interval].at<float>(r, c);

	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++)
				if (intensity
						< dog_pyr[octave][interval + i].at<float>(r + j, c + k))
					return false;
	return true;
}

/**
 * Parse a given input into a mathematical expression or define a new
 * variable or assign value to a variable
 *
 * @param expression
 *            the string to check
 * @return result of the induced calculation
 */
bool isMinimum(vector<vector<Mat> >& dog_pyr, int octave, int interval, int r,
		int c) {
	float intensity = dog_pyr[octave][interval].at<float>(r, c);

	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++)
				if (intensity
						> dog_pyr[octave][interval + i].at<float>(r + j, c + k))
					return false;
	return true;
}

/**
 * Parse a given input into a mathematical expression or define a new
 * variable or assign value to a variable
 *
 * @param expression
 *            the string to check
 * @return result of the induced calculation
 */
bool isExtrema(vector<vector<Mat> >& dog_pyr, int octave, int interval, int r,
		int c) {

	return isMaximum(dog_pyr, octave, interval, r, c)
			|| isMinimum(dog_pyr, octave, interval, r, c);
}

/**
 * Parse a given input into a mathematical expression or define a new
 * variable or assign value to a variable
 *
 * @param expression
 *            the string to check
 * @return result of the induced calculation
 */
void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr,
		vector<KeyPoint>& keypoints) {
	int octvs = dog_pyr.size();
	int intvls = dog_pyr[0].size();

	for (int o = 0; o < octvs; o++)
		for (int i = 1; i <= intvls; i++)
			for (int r = SIFT_IMG_BORDER;
					r < dog_pyr[o][0].rows - SIFT_IMG_BORDER; r++)
				for (int c = SIFT_IMG_BORDER;
						c < dog_pyr[o][0].cols - SIFT_IMG_BORDER; c++) {
					if (isExtrema(dog_pyr, o, i, r, c))
						keypoints.push_back(KeyPoint(r, c, 0, -1, 0, o));
				}
}

/**
 * Parse a given input into a mathematical expression or define a new
 * variable or assign value to a variable
 *
 * @param expression
 *            the string to check
 * @return result of the induced calculation
 */
void cleanPoints(vector<KeyPoint>& keypoints, Mat& image, int curv_thr) {
	float rx, ry, fxx, fxy, fyy, deter;
	float trace, curvature;
	bool skipPoint;
	vector<KeyPoint> cleanpoints;

	for (int i = 0; i < keypoints.size(); i++) {
		KeyPoint pnt = keypoints[i];
		skipPoint = false;

		// Low Contrast
		if (abs(image.at<float>(pnt.pt)) < CONTRAST_THRESHOLD) {
			// reject_contrast_count++;
			skipPoint = true;
		} else {
			rx = pnt.pt.x;
			ry = pnt.pt.y;

			// Get the elements of the 2x2 Hessian Matrix
			fxx = image.at<float>(rx - 1, ry) + image.at<float>(rx + 1, ry)
					- 2 * image.at<float>(rx, ry); // 2nd order derivate in x direction
			fyy = image.at<float>(rx, ry - 1) + image.at<float>(rx, ry + 1)
					- 2 * image.at<float>(rx, ry); // 2nd order derivate in y direction
			fxy = image.at<float>(rx - 1, ry - 1)
					+ image.at<float>(rx + 1, ry + 1)
					- image.at<float>(rx - 1, ry + 1)
					- image.at<float>(rx + 1, ry - 1);
			// Partial derivate in x and y direction
			// Find Trace and Determinant of this Hessian

			trace = (float) (fxx + fyy);
			deter = (fxx * fyy) - (fxy * fxy);
			curvature = (float) (trace * trace / deter);

			// Reject edge points if curvature condition is not satisfied
			if (deter < 0 || curvature > curv_thr) {
				// reject_contrast_count++;
				skipPoint = true;
			}
		}

		if (!skipPoint)
			cleanpoints.push_back(pnt);
	}

	keypoints = cleanpoints;
}

