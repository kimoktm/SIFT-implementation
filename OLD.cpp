#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <math.h>
#include <stdio.h>

using namespace std;
using namespace cv;

#define SIFT_IMG_BORDER 			10
#define CONTRAST_THRESHOLD			0.03
#define R_CURVATURE             	10.0
#define PI							3.14159265358979323846

vector<vector<Mat> > dogpyr;
vector<vector<Mat> > pyr;
vector<Mat> keypointsGradients;
vector<Mat> keypointsMagnitudes;
int nOctaves;
int gImages;
int histogramMargin;
int halfMargin;
int DOGImages;
double contrastThreshold;
double initialsigma;
double sigma;
vector<vector<double> > descriptorAllFeatures;

void initialization() {
	nOctaves = 4;
	gImages = 5 + 3;
	histogramMargin = 16;
	halfMargin = histogramMargin / 2;
	DOGImages = gImages - 1;
	initialsigma = sqrt(2.0) / 2;
	sigma = initialsigma;
}

Mat downSample(Mat& image) {
	Mat gauss;
	GaussianBlur(image, gauss, Size(0, 0), sqrt(2) / 2, 0);

	Mat temp = Mat(Size(gauss.cols / 2, gauss.rows), image.type());
	for (int i = 0; i < temp.cols; i++)
		gauss.col(i * 2).copyTo(temp.col(i));

	Mat dest = Mat(Size(temp.cols, temp.rows / 2), image.type());
	for (int i = 0; i < dest.rows; i++)
		temp.row(i * 2).copyTo(dest.row(i));

	return dest;
}

void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& gauss_pyr, int nOctaves) {
	double sigma;
	Mat tempImage;
	image.copyTo(tempImage);

	for (int i = 0; i < nOctaves; i++) {
		sigma = initialsigma;
		vector<Mat> pyr_intervals;

		for (int j = 0; j < gImages; j++) {
			Mat blurredImage;
			GaussianBlur(tempImage, blurredImage, Size(0, 0), sigma, 0);
			pyr_intervals.push_back(blurredImage);
			blurredImage.release();
			sigma *= sqrt(2);
		}

		gauss_pyr.push_back(pyr_intervals);
		tempImage = downSample(tempImage);
	}

	tempImage.release();
}

vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr) {
	vector<vector<Mat> > dog_pyr;

	for (int i = 0; i < nOctaves; i++) {
		vector<Mat> dog_intervals;

		for (int j = 0; j < gImages - 1; j++) {
			dog_intervals.push_back(gauss_pyr[i][j] - gauss_pyr[i][j + 1]);
		}

		dog_pyr.push_back(dog_intervals);
	}

	return dog_pyr;
}

void maxInHistogram(vector<double> histogram, int &maximum, int &secondmax, int &indexMax, int &indexSecond) {

	maximum = histogram[0];
	secondmax = histogram[0];
	indexMax = 0;
	indexSecond = 0;

	for (size_t i = 0; i < histogram.size(); i++) {
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

void computeGradient(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& features) {
	int range = 10;
	int maximum = 360;

	for (size_t z = 0; z < features.size(); z++) {
		Mat image = dog_pyr[features[z].octave][features[z].size];
		int keyx = features[z].pt.x;
		int keyy = features[z].pt.y;

		if (keyx - halfMargin - 1 < 0 || keyx + halfMargin + 1 > image.cols || keyy - halfMargin - 1 < 0
				|| keyy + halfMargin + 1 > image.rows) {
			return;
		} else {

			Mat tempMagnitude = (Mat_<float>(histogramMargin, histogramMargin));
			Mat tempGradient = (Mat_<float>(histogramMargin, histogramMargin));
			for (int i = 0; i < histogramMargin; i++) {

				for (int j = 0; j < histogramMargin; j++) {
					float diffx, diffy, magnitude, gradient;

					diffx = image.at<float>(keyx + i + 1 - halfMargin, keyy - halfMargin + j)
							- image.at<float>(keyx + i - 1 - halfMargin, keyy - halfMargin + j);
					diffy = image.at<float>(keyx - halfMargin + i, keyy + j + 1 - halfMargin)
							- image.at<float>(keyx - halfMargin + i, keyy + j - 1 - halfMargin);
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

void computeDescriptors() {
	for (size_t points = 0; points < keypointsGradients.size(); points++) {
		Mat temp = keypointsGradients[points];
		vector<double> singleDescriptor;
		singleDescriptor.reserve(128);
		for (int xBlock = 0; xBlock < temp.cols; xBlock += 4) {

			for (int yBlock = 0; yBlock < 16; yBlock += 4) {
				Mat blockMatrix = Mat::zeros(4, 4, CV_32F);
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						blockMatrix.at<float>(i, j) = temp.at<float>(xBlock + i, yBlock + j);

					}
				}
				vector<double> singleHistogram = histogramize(blockMatrix, 45, 360);
				singleDescriptor.insert(singleDescriptor.end(), singleHistogram.begin(), singleHistogram.end());

			}
		}
		descriptorAllFeatures.push_back(singleDescriptor);

	}
}

bool isExtrema(vector<vector<Mat> >& dog_pyr, int octave, int interval, int r, int c) {
	float intensity = dog_pyr[octave][interval].at<float>(r, c);

	if (intensity > 0) {
		for (int i = -1; i <= 1; i++)
			for (int j = -1; j <= 1; j++)
				for (int k = -1; k <= 1; k++)
					if (intensity <= dog_pyr[octave][interval + i].at<float>(r + j, c + k) && (j != 0 || k != 0))
						return false;
	} else {
		for (int i = -1; i <= 1; i++)
			for (int j = -1; j <= 1; j++)
				for (int k = -1; k <= 1; k++)
					if (intensity >= dog_pyr[octave][interval + i].at<float>(r + j, c + k) && (j != 0 || k != 0))
						return false;
	}
	return true;
}

bool cleanPoints(Point loc, Mat& image, int curv_thr) {
	float rx, ry, fxx, fxy, fyy, deter;
	float trace, curvature;

	if (abs(image.at<float>(loc)) < CONTRAST_THRESHOLD) {
		return false;
	} else {
		rx = loc.x;
		ry = loc.y;

		fxx = image.at<float>(rx - 1, ry) + image.at<float>(rx + 1, ry) - 2 * image.at<float>(rx, ry); // 2nd order derivate in x direction
		fyy = image.at<float>(rx, ry - 1) + image.at<float>(rx, ry + 1) - 2 * image.at<float>(rx, ry); // 2nd order derivate in y direction
		fxy = image.at<float>(rx - 1, ry - 1) + image.at<float>(rx + 1, ry + 1) - image.at<float>(rx - 1, ry + 1)
				- image.at<float>(rx + 1, ry - 1);

		trace = fxx + fyy;
		deter = fxx * fyy - fxy * fxy;
		curvature = trace * trace / deter;

		if (deter < -0.00001 || curvature > curv_thr) {
			return false;
		}
	}

	return true;
}

void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints) {
	int octvs = dog_pyr.size();
	int intvls = dog_pyr[0].size() - 2;

	for (int o = 0; o < octvs; o++) {
		for (int i = 1; i <= intvls; i++) {
			for (int r = SIFT_IMG_BORDER; r < dog_pyr[o][0].rows - SIFT_IMG_BORDER; r++) {
				for (int c = SIFT_IMG_BORDER; c < dog_pyr[o][0].cols - SIFT_IMG_BORDER; c++) {
					if (isExtrema(dog_pyr, o, i, r, c)) {
						if (cleanPoints(Point(c, r), dog_pyr[o][i],
						R_CURVATURE))
							keypoints.push_back(KeyPoint(c, r, i, -1, 0, o));
					}
				}
			}
		}
	}
}

void drawKeyPoints(vector<KeyPoint>& keypoints, Mat& image) {
	for (size_t i = 0; i < keypoints.size(); i++) {
		Point pt1 = keypoints[i].pt;
		Point pt2;
		pt2.x = pt1.x + cos(keypoints[i].angle * PI / 180.0f) * pow(2, keypoints[i].octave);
		pt2.y = pt1.y + sin(keypoints[i].angle * PI / 180.0f) * pow(2, keypoints[i].octave);
//		line(image, pt1 * pow(2, keypoints[i].octave),
//				pt2 * pow(2, keypoints[i].octave), Scalar(150, 0, 0), 0.5,
//				CV_AA);

		circle(image, pt1 * pow(2, keypoints[i].octave), 3, Scalar(80, 80, 80), -1, CV_AA);
		circle(image, pt1 * pow(2, keypoints[i].octave), 3, Scalar(0, 69, 255), -1, CV_AA);
//		circle(image, pt2, 1, Scalar(255, 255, 255));
	}
	imshow("SIFT", image);
}

int main(int argc, char** argv) {
	Mat image, image2;
	Mat imageColor = imread("../Test-Data/images/cluster150.png", CV_LOAD_IMAGE_COLOR);

	if (!imageColor.data) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvtColor(imageColor, image, CV_BGR2GRAY);
	cvtColor(imageColor, image2, CV_BGR2GRAY);
	normalize(image, image, 0, 1, NORM_MINMAX, CV_32F);

	initialization();
	buildGaussianPyramid(image, pyr, nOctaves);
	dogpyr = buildDogPyr(pyr);

	vector<KeyPoint> keypoints;
	getScaleSpaceExtrema(dogpyr, keypoints);
	computeGradient(dogpyr, keypoints);
	cout << "SIZE: " << keypoints.size() << endl;
	computeDescriptors();
	cout << descriptorAllFeatures.size() << endl;
	drawKeyPoints(keypoints, imageColor);

	// OPENCV SIFT //
	SiftFeatureDetector detector;
	vector<cv::KeyPoint> siftkeypoints;
	detector.detect(image2, siftkeypoints);

	cv::Mat output;
	cv::drawKeypoints(image2, siftkeypoints, output);
	cv::imshow("sift_result", output);

	cout << "---------------------------" << endl;
	waitKey(0);
	return 0;
}
