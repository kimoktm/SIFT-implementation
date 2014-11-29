/*
 * SIFT.h
 *
 *  Created on: Nov 18, 2014
 *  Author: Ahmed, Karim, Mostafa
 */

#ifndef SIFT_H
#define SIFT_H

#include <math.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#define SIFT_INTVLS							5
#define SIFT_OCTVES							4
#define SIFT_IMG_BORDER						10
#define SIFT_HIST_BOREDER					8
#define SIFT_CURV_THR						10
#define SIFT_CONTR_THR						0.03
#define SIFT_DETER_THR						0
#define INTERPOLATION_SIGMA					0.707106781
#define SIFT_INIT_SIGMA						0.707106781
#define SIFT_STEP_SIGMA						1.414213562
#define PI									3.141592653

using namespace std;
using namespace cv;

class SIFT
{
private:
	vector<Mat> keypointsGradients;
	vector<Mat> keypointsMagnitudes;

	/** Convert a given angle from radians to degrees **/
	double deg2rad(float deg);

	/** Convert a given angle from degrees to radians **/
	double rad2deg(float rad);

	/** Downsamples an image to quarter its size **/
	Mat downSample(Mat& image);

	/** Gets the first and its index in a given histogram **/ 
	void histogramMax(vector<double> histogram, int &maximum, int &indexMax);

	/** Tests if the given point is an extrema by comparing it to it's surroundings **/ 	
	bool isExtrema(vector<vector<Mat> >& dog_pyr, int octave, int interval, int r, int c);

	/** Build a gradient histogram from the given window and range **/
	vector<double> buildHistogram(Mat matrix, int range, int maximum);

public:
	/** Finds the SIFT keypoints in a given image **/
	void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints,
			int nOctaves = SIFT_OCTVES, int nIntervals = SIFT_INTVLS);

	/** Build Scale Space guassian pyramid from an image **/
	void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, 
		int nOctaves = SIFT_OCTVES, int nIntervals = SIFT_INTVLS);

	/** Tests if the given point is a good feature **/
	bool cleanPoints(Point position, Mat& image, int curv_thr = SIFT_CURV_THR,
			float cont_thr = SIFT_CONTR_THR, float dtr_thr = SIFT_DETER_THR);

	/** Gets the extremas from the DOG pyramid **/
	void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints,
		int curv_thr = SIFT_CURV_THR);
	
	/** Compute the Orientation histogram for the DOG pyramid **/
	void computeOrientationHist(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);
	
	/** Draws the given keypoints on the given image **/
	void drawKeyPoints(Mat& image, vector<KeyPoint>& keypoints);

	/** Finds the SIFT keypoints in a given image **/
	vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);
	
	/** Compute the SIFT descriptor of each keypoints **/
	vector<vector<double> > computeDescriptors();
};

#endif

