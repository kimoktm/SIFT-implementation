#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/**
 * Detrimes if a pixel is maximum in
 *
 * @param expression
 *            the string to check
 * @return result of the induced calculation
 */
bool isMaximum(vector<vector<Mat> >& dog_pyr, int octave, int interval, int x,
		int y) {
	float intensity = dog_pyr[octave][interval].at<float>(x, y);

	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++)
				if (intensity
						< dog_pyr[octave][interval + i].at<float>(x + j, y + k))
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
bool isMinimum(vector<vector<Mat> >& dog_pyr, int octave, int interval, int x,
		int y) {
	float intensity = dog_pyr[octave][interval].at<float>(x, y);

	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++)
				if (intensity
						> dog_pyr[octave][interval + i].at<float>(x + j, y + k))
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
bool isExtrema(vector<vector<Mat> >& dog_pyr, int octave, int interval, int x,
		int y) {

	return isMaximum(dog_pyr, octave, interval, x, y)
			|| isMinimum(dog_pyr, octave, interval, x, y);
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

}
