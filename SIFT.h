/*
 * main.cpp
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
#define SIFT_IMG_BORDER 					10
#define SIFT_HIST_BOREDER					8
#define SIFT_CURV_THR						10
#define SIFT_CONTR_THR						0.03
#define SIFT_DETER_THR						-0.00001
#define INTERPOLATION_SIGMA					0.707106781
#define SIFT_INIT_SIGMA						0.707106781
#define SIFT_STEP_SIGMA						1.414213562
#define PI                          		3.141592653

using namespace std;
using namespace cv;

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints, int nOctaves = 4);
void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves);
bool cleanPoints(Point position, Mat& image, int curv_thr);
Mat downSample(Mat& image);
vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr);
vector<Mat> computeOrientationHist(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);
void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints);
void drawKeyPoints(Mat& image, vector<KeyPoint>& keypoints);

#endif
