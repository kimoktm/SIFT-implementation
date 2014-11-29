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
#define SIFT_IMG_BORDER 					10
#define SIFT_HIST_BOREDER					8
#define SIFT_CURV_THR						10
#define SIFT_CONTR_THR						0.03
#define SIFT_DETER_THR						0
#define INTERPOLATION_SIGMA					0.707106781
#define SIFT_INIT_SIGMA						0.707106781
#define SIFT_STEP_SIGMA						1.414213562
#define PI                          		3.141592653

using namespace std;
using namespace cv;

class SIFT
{
private:
	vector<Mat> keypointsGradients;
	vector<Mat> keypointsMagnitudes;

	double deg2rad(float deg);
	double rad2deg(float rad);
	Mat downSample(Mat& image);

	void histogramMax(vector<double> histogram, int &maximum, int &indexMax);
	bool isExtrema(vector<vector<Mat> >& dog_pyr, int octave, int interval, int r, int c);
	bool cleanPoints(Point position, Mat& image, int curv_thr = SIFT_CURV_THR,
			float cont_thr = SIFT_CONTR_THR, float dtr_thr = SIFT_DETER_THR);

	vector<double> buildHistogram(Mat matrix, int range, int maximum);
public:
	void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints,
			int nOctaves = SIFT_OCTVES, int nIntervals = SIFT_INTVLS);
	void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves, int nIntervals);
	void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);
	void computeOrientationHist(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);
	void drawKeyPoints(Mat& image, vector<KeyPoint>& keypoints);

	vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);
	vector<vector<double> > computeDescriptors();
};

#endif

