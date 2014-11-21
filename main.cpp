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
# define pi           3.14159265358979323846  /* pi */

using namespace std;
using namespace cv;

//keep track of all guassian images and DOGs
vector<vector<Mat> > dogpyr;
vector<vector<Mat> > pyr;
vector<Mat> keypointsGradients;
vector<vector<double> > descriptorAllFeatures;
vector<Mat> keypointsMagnitudes;
int nOctaves;
int gImages;
int histogramMargin;
int halfMargin;
int DOGImages;
//define thresholds
double contrastThreshold;
double curvatureThreshold;
//define starting sigma for guassian
double initialsigma;
double sigma;

void initialization() {
	nOctaves = 4;
	gImages = 5;
	histogramMargin = 16;
	halfMargin = histogramMargin / 2;
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

void maxInHistogram(vector<double> histogram, int &maximum, int &secondmax,
		int &indexMax, int &indexSecond) {

	maximum = histogram[0];
	secondmax = histogram[0];
	indexMax = 0;
	indexSecond = 0;

	for (int i = 0; i < histogram.size(); i++) {
		if (maximum < histogram[i]) {
			secondmax = maximum;
			indexSecond = indexMax;

			maximum = histogram[i];
			indexMax = i;
		}
	}

}

vector<double> histogramize(Mat matrix, int range, int maximum) {
	int size = maximum / range;
	vector<double> histo(size);
	for (int i = 0; i < histo.size(); i++) {
		histo[i] = 0;
	}
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			int index = matrix.at<float>(i, j) / range;
			histo[index]++;
		}
	}
	return histo;

}
//keypoint index and image

void computeDescriptors() {
	for (int points = 0; points < keypointsGradients.size(); points++) {
		Mat temp = keypointsGradients[points];
		vector<double> singleDescriptor;
		singleDescriptor.reserve(128);
		for (int xBlock = 0; xBlock < temp.cols; xBlock += 4) {

			for (int yBlock = 0; yBlock < 16; yBlock += 4) {
				Mat blockMatrix = Mat::zeros(4, 4, CV_32F);
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						blockMatrix.at<float>(i, j) = temp.at<float>(xBlock + i,
								yBlock + j);

					}
				}
				vector<double> singleHistogram = histogramize(blockMatrix, 45,
						360);
				singleDescriptor.insert(singleDescriptor.end(),
						singleHistogram.begin(), singleHistogram.end());

			}
		}

		descriptorAllFeatures.push_back(singleDescriptor);

	}

}

void computeGradient(Mat image, vector<KeyPoint> features) {
//ignore edges
	int range = 10;
	int maximum = 360;

	for (int z = 0; z < features.size(); z++) {

		int keyx = features[z].pt.x;
		int keyy = features[z].pt.y;

		if (keyx - halfMargin - 1 < 0 || keyx + halfMargin + 1 > image.cols
				|| keyy - halfMargin - 1 < 0
				|| keyy + halfMargin + 1 > image.rows) {
			return;
		} else {

			Mat tempMagnitude = (Mat_<float>(histogramMargin, histogramMargin));
			Mat tempGradient = (Mat_<float>(histogramMargin, histogramMargin));
			for (int i = 0; i < histogramMargin; i++) {

				for (int j = 0; j < histogramMargin; j++) {
					float diffx, diffy, magnitude, gradient;

					diffx = image.at<float>(keyx + i + 1 - halfMargin,
							keyy - halfMargin + j)
							- image.at<float>(keyx + i - 1 - halfMargin,
									keyy - halfMargin + j);
					diffy = image.at<float>(keyx - halfMargin + i,
							keyy + j + 1 - halfMargin)
							- image.at<float>(keyx - halfMargin + i,
									keyy + j - 1 - halfMargin);
					magnitude = sqrt(pow(diffx, 2) + pow(diffy, 2));
					gradient = atan2f(diffy, diffx);

					if (gradient < 0) {
						gradient += (2 * pi);
					}
					gradient *= 360 / (2 * pi);

					tempMagnitude.at<float>(i, j) = magnitude;
					tempGradient.at<float>(i, j) = gradient;

				}

			}

			keypointsGradients.push_back(tempGradient);
			keypointsMagnitudes.push_back(tempMagnitude);

			int maxima, secondmax, indexMax, indexSecond;

			vector<double> histo = histogramize(tempGradient, range, maximum);

			maxInHistogram(histo, maxima, secondmax, indexMax, indexSecond);

			int angleOrientation = indexMax * range;
			angleOrientation = angleOrientation + (range / 2);
			features[z].angle = angleOrientation;
		}
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
	KeyPoint testing(9, 9, 0, 0, 0, 2);
	vector<KeyPoint> booya;
	booya.push_back(testing);
	computeGradient(image, booya);
	computeDescriptors();
	vector<vector<Mat> > pyr;
	vector<Mat> intt;
	intt.push_back(C);
	intt.push_back(D);
	intt.push_back(E);
	pyr.push_back(intt);

	//cout << C << endl;
	//cout << "=======================================" << endl;
	//cout << D << endl;
	//cout << "=======================================" << endl;
	//cout << E << endl;
	//cout << "=======================================" << endl;

	waitKey(0);
	return 0;
}
