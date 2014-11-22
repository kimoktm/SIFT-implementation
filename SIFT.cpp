#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 			10
#define CONTRAST_THRESHOLD			0.03
#define R_CURVATURE             	10.0
#define PI							3.14159265358979323846

//keep track of all guassian images and DOGs
vector<vector<Mat> > dogpyr;
vector<vector<Mat> > pyr;
vector<Mat> keypointsGradients;
vector<Mat> keypointsMagnitudes;
int nOctaves;
int gImages;
int histogramMargin;
int halfMargin;
int DOGImages;
//define thresholds
double contrastThreshold;
//double curvatureThreshold;
//define starting sigma for guassian
double initialsigma;
double sigma;

/************************************/
void initialization() {
	nOctaves = 4;
	gImages = 5 + 3;
	histogramMargin = 16;
	halfMargin = histogramMargin / 2;
	DOGImages = gImages - 1;
	initialsigma = sqrt(2);
	sigma = initialsigma;
}

Mat downSample(Mat& image) {
//down sample the image half the size for the next octave

	Mat gauss;
//	image.copyTo(gauss);
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

void buildGaussianPyramid(Mat image, vector<vector<Mat> >& pyr, int nOctaves) {

	for (int i = 0; i < nOctaves; i++) {
		sigma = initialsigma;
		vector<Mat> allGuassians;
		//applies guassian filter on image
		for (int j = 0; j < gImages; j++) {
//			if (j == 0) {
//				//leaves the 1st image without blur and pushes in global vector
//				allGuassians.push_back(image);
//			} else {
			Mat blurredImage;
			GaussianBlur(image, blurredImage, Size(0, 0), sigma, 0);
			allGuassians.push_back(blurredImage);
			//increases blur
			sigma *= sqrt(2);
//			}
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
//			absdiff(gauss_pyr[i][j], gauss_pyr[i][j + 1], DOG);
			DOG = gauss_pyr[i][j] - gauss_pyr[i][j + 1];
			allDOGs.push_back(DOG);
		}
		dogpyramid.push_back(allDOGs);
	}
	return dogpyramid;
}

/************************************/

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

void computeGradient(vector<vector<Mat> >& dog_pyr,
		vector<KeyPoint>& features) {
//ignore edges
	int range = 10;
	int maximum = 360;

	for (int z = 0; z < features.size(); z++) {
		Mat image = dog_pyr[features[z].octave][features[z].size];
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
						gradient += (2 * PI);
					}
					gradient *= 360 / (2 * PI);

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

//	return isMaximum(dog_pyr, octave, interval, r, c)
//			|| isMinimum(dog_pyr, octave, interval, r, c);

	float intensity = dog_pyr[octave][interval].at<float>(r, c);

	if (intensity > 0) {
		for (int i = -1; i <= 1; i++)
			for (int j = -1; j <= 1; j++)
				for (int k = -1; k <= 1; k++)
					if (intensity
							<= dog_pyr[octave][interval + i].at<float>(r + j,
									c + k) && (j != 0 || k != 0))
						return false;
	} else {
		for (int i = -1; i <= 1; i++)
			for (int j = -1; j <= 1; j++)
				for (int k = -1; k <= 1; k++)
					if (intensity
							>= dog_pyr[octave][interval + i].at<float>(r + j,
									c + k) && (j != 0 || k != 0))
						return false;
	}
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
bool cleanPoints(Point loc, Mat& image, int curv_thr) {
	float rx, ry, fxx, fxy, fyy, deter;
	float trace, curvature;

	// Low Contrast
	if (abs(image.at<float>(loc)) < CONTRAST_THRESHOLD) {
		// reject_contrast_count++;
		return false;
	} else {
		rx = loc.x;
		ry = loc.y;

		// Get the elements of the 2x2 Hessian Matrix
		fxx = image.at<float>(rx - 1, ry) + image.at<float>(rx + 1, ry)
				- 2 * image.at<float>(rx, ry); // 2nd order derivate in x direction
		fyy = image.at<float>(rx, ry - 1) + image.at<float>(rx, ry + 1)
				- 2 * image.at<float>(rx, ry); // 2nd order derivate in y direction
		fxy = image.at<float>(rx - 1, ry - 1) + image.at<float>(rx + 1, ry + 1)
				- image.at<float>(rx - 1, ry + 1)
				- image.at<float>(rx + 1, ry - 1);
		// Partial derivate in x and y direction
		// Find Trace and Determinant of this Hessian

		trace = (float) (fxx + fyy);
		deter = (fxx * fyy) - (fxy * fxy);
		curvature = (float) (trace * trace / deter);

//		cout << curvature << endl;
		// Reject edge points if curvature condition is not satisfied
//		if (deter < 0 || curvature > curv_thr) {
		if (curvature > curv_thr) {

			// reject_contrast_count++;
			return false;
		}
	}

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
void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr,
		vector<KeyPoint>& keypoints) {
	int octvs = dog_pyr.size();
	int intvls = dog_pyr[0].size() - 3;

	for (int o = 0; o < octvs; o++) {
		for (int i = 1; i <= intvls; i++) {
			for (int c = SIFT_IMG_BORDER;
					c < dog_pyr[o][0].cols - SIFT_IMG_BORDER; c++) {
				for (int r = SIFT_IMG_BORDER;
						r < dog_pyr[o][0].rows - SIFT_IMG_BORDER; r++) {
					if (isExtrema(dog_pyr, o, i, r, c)) {
						if (cleanPoints(Point(c, r), dog_pyr[o][i],
						R_CURVATURE))
							keypoints.push_back(KeyPoint(c, r, i, -1, 0, o));
//						cout << r << " , " << c << endl;
					}
				}
			}
		}
	}
}

Scalar colors[] = { Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 255, 0),
		Scalar(125, 125, 0), Scalar(0, 125, 125) };

void drawKeyPoints(vector<KeyPoint>& keypoints, Mat& image) {
	for (int i = 0; i < keypoints.size(); i++) {
		Point pt1 = keypoints[i].pt;
		Point pt2;
		pt2.x = pt1.x
				+ cos(keypoints[i].angle * PI / 180.0f)
						* pow(2, keypoints[i].octave);
		pt2.y = pt1.y
				+ sin(keypoints[i].angle * PI / 180.0f)
						* pow(2, keypoints[i].octave) * 15;
//		line(image, pt1 * pow(2, keypoints[i].octave),
//				pt2 * pow(2, keypoints[i].octave), Scalar(150, 0, 0), 2);
		circle(image, pt1 * pow(2, keypoints[i].octave), 3, colors[2], -1);

//		circle(image, pt2, 1, Scalar(255, 255, 255));
	}
	imshow("SIFT", image);
}

int main(int argc, char** argv) {
	Mat image, image2;
	Mat imageColor = imread("../Test-Data/images/cluster150.png",
			CV_LOAD_IMAGE_COLOR);

	if (!imageColor.data) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

//	resize(imageColor, imageColor,
//			Size(imageColor.cols / 2, imageColor.rows / 2));
//	normalize image and define octave numbers and guassian images to produce
	cvtColor(imageColor, image, CV_BGR2GRAY);
	cvtColor(imageColor, image2, CV_BGR2GRAY);

	normalize(image, image, 0, 1, NORM_MINMAX, CV_32F);
	imshow("BG", image);

	initialization();
	buildGaussianPyramid(image, pyr, nOctaves);
	dogpyr = buildDogPyr(pyr);

	vector<KeyPoint> keypoints;
	getScaleSpaceExtrema(dogpyr, keypoints);
	computeGradient(dogpyr, keypoints);
	cout << "SIZE: " << keypoints.size() << endl;

	drawKeyPoints(keypoints, imageColor);

	SiftFeatureDetector detector;
	vector<cv::KeyPoint> siftkeypoints;
	detector.detect(image2, siftkeypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(image2, siftkeypoints, output);
	cv::imshow("sift_result", output);

	cout << "---------------------------" << endl;
	waitKey(0);
	return 0;
}
