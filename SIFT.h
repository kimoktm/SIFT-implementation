/*
 * main.cpp
 *
 *  Created on: Nov 18, 2014
 *      Author: Ahmed, Karim, Mostafa
 */

#ifndef SIFT_H
#define SIFT_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <stdio.h>

#define SIFT_INTVLS							5
#define SIFT_IMG_BORDER 					10
#define SIFT_CURV_THR						10
#define SIFT_CONTR_THR						0.03
#define SIFT_DETER_THR						-0.00001
#define INTERPOLATION_SIGMA					0.7071067812
#define SIFT_INIT_SIGMA						0.7071067812
#define SIFT_STEP_SIGMA						1.414213562

using namespace std;
using namespace cv;

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints);
void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves);
void cleanPoints(Mat& image, int curv_thr);
Mat downSample(Mat& image);
vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);
vector<double> computeOrientationHist(const Mat& image);
void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);

#endif
