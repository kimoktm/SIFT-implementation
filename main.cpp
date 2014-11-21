/*
 * main.cpp
 *
 *  Created on: Nov 18, 2014
 *      Author: Ahmed, Karim, Mostafa
 */

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

//keep track of all guassian images and DOGs
vector<vector<Mat> > dogpyr;
vector<vector<Mat> > pyr;
vector<Mat> keypointsGradients;
vector<Mat> keypointsMagnitudes;
int nOctaves;
int gImages;
int DOGImages;
//define thresholds
double contrastThreshold;
double curvatureThreshold;
//define starting sigma for guassian
double initialsigma;
double sigma;

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

bool isExtrema(vector<vector<Mat> >& dog_pyr, int octave, int interval, int x,
		int y) {

	return isMaximum(dog_pyr, octave, interval, x, y)
			|| isMinimum(dog_pyr, octave, interval, x, y);
}

void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr,
		vector<KeyPoint>& keypoints) {

}

void initialization() {
	nOctaves = 4;
	gImages = 5;
	DOGImages = gImages - 1;
	contrastThreshold = 0.03;
	curvatureThreshold = 10.0;
	initialsigma = sqrt(2);
	sigma = initialsigma;
}

Mat downSample(Mat& image) {
//down sample the image half the size for the next octave

	Mat gauss;
	GaussianBlur(image, gauss, Size(0, 0), sqrt(2) / 2, 0);

//Downsample columns and save it to temp
	Mat temp = Mat(Size(gauss.cols / 2, gauss.rows), image.type());
	for (int i = 0; i < temp.cols; i++)
		gauss.col(i * 2).copyTo(temp.col(i));

//Downsample rows and return it
	Mat dest = Mat(Size(temp.cols, temp.rows / 2), image.type());
	for (int i = 0; i < dest.rows; i++)
		temp.row(i * 2).copyTo(dest.row(i));

	return dest;
}

void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& pyr, int nOctaves) {

	for (int i = 0; i < nOctaves; i++) {
		sigma = initialsigma;
		vector<Mat> allGuassians;
		//applies guassian filter on image
		for (int j = 0; j < gImages; j++) {

			if (j == 0) {
				//leaves the 1st image without blur and pushes in global vector
				allGuassians.push_back(image);
			} else {
				Mat blurredImage;
				GaussianBlur(image, blurredImage, Size(0, 0), sigma, 0);
				allGuassians.push_back(blurredImage);
				//increases blur
				sigma *= sqrt(2);
			}
		}
		pyr.push_back(allGuassians);
		image = downSample(image);
	}
}

vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr) {
	vector<vector<Mat> > dogpyramid;
	for (int i = 0; i < nOctaves; i++) {
		//gets DOG across guassian images
		vector<Mat> allDOGs;
		for (int j = 0; j < DOGImages; j++) {
			Mat DOG;
			absdiff(gauss_pyr[i][j], gauss_pyr[i][j + 1], DOG);
			allDOGs.push_back(DOG);
		}
		dogpyramid.push_back(allDOGs);
	}
	return dogpyramid;
}

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints) {

}

//based on contrast //and principal curvature ratio
void cleanPoints(Mat& image, int curv_thr) {

}

// Calculates the gradient vector of the feature
vector<double> computeOrientationHist(const Mat& image) {

}

//keypoint index and image


void computeGradient(Mat image, int keyx, int keyy) {
//ignore edges
	if (keyx - 9 < 0 || keyx + 9 > image.cols || keyy - 9 < 0
			|| keyy + 9 > image.rows) {
		return;
	} else {
		Mat tempMagnitude = (Mat_<float>(16, 16));
		Mat tempGradient = (Mat_<float>(16, 16));
		for (int i = 0; i < 16; i++) {

			for (int j = 0; j < 16; j++) {
				float diffx, diffy, magnitude, gradient;

				diffx = image.at<float>(keyx + i + 1 - 8, keyy - 8 + j)
						- image.at<float>(keyx + i - 1 - 8, keyy - 8 + j);
				diffy = image.at<float>(keyx - 8 + i, keyy + j + 1 - 8)
						- image.at<float>(keyx - 8 + i, keyy + j - 1 - 8);
				magnitude = sqrt(pow(diffx, 2) + pow(diffy, 2));
				gradient = atan(diffy / diffx);

				tempMagnitude.at<float>(i, j) = magnitude;
				tempGradient.at<float>(i, j) = gradient;

			}

		}

		keypointsGradients.push_back(tempGradient);
		keypointsMagnitudes.push_back(tempMagnitude);
	}

}

int main(int argc, char** argv) {

	Mat image;
	//karim linux
	//image = imread("../Test-Data/images/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//jihad windows
	image = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//shawky apple
	//image = imread("../Test-Data/images/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	if (!image.data) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

//normalize image and define octave numbers and guassian images to produce
	//cvtColor(image, image, CV_BGR2GRAY);
	normalize(image, image, 0, 1, NORM_MINMAX, CV_32F);

	initialization();

	buildGaussianPyramid(image, pyr, nOctaves);

	dogpyr = buildDogPyr(pyr);

//////////////////////////////////////////////////////////////////////////
//testing output
//	int count = 0;
//	for (int i = 0; i < dogpyr.size(); i++) {
//		for (int y = 0; y < dogpyr[i].size(); y++) {
//			count++;
//			imshow("et3'adena eh enahrda ya shabab" + (count), dogpyr[i][y]);
//		}
//	}
//////////////////////////////////////////////////////////////////////////
	Mat C = (Mat_<double>(3, 3) << 0, 1, 0, 1, 5, -1, 0, -1, 0);
	Mat D = (Mat_<double>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0);
	Mat E = (Mat_<double>(3, 3) << 13, 20, 52, 34, 77, 89, 54, 20, 46);
	normalize(C, C, 0, 1, NORM_MINMAX, CV_32F);
	normalize(E, E, 0, 1, NORM_MINMAX, CV_32F);
	normalize(D, D, 0, 1, NORM_MINMAX, CV_32F);
	computeGradient(image, 20, 20);
	vector<vector<Mat> > pyr;
	vector<Mat> intt;
	intt.push_back(C);
	intt.push_back(D);
	intt.push_back(E);
	pyr.push_back(intt);

	cout << C << endl;
	cout << "=======================================" << endl;
	cout << D << endl;
	cout << "=======================================" << endl;
	cout << E << endl;
	cout << "=======================================" << endl;

	bool ext = isMinimum(pyr, 0, 1, 1, 1);
	cout << ext << endl;

	waitKey(0);
	return 0;
}
