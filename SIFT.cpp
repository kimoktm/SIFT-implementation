#include "SIFT.h"

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints, int nOctaves)
{
	Mat _image;
	cvtColor(image, _image, CV_BGR2GRAY);
	normalize(_image, _image, 0, 1, NORM_MINMAX, CV_32F);

	vector<vector<Mat> > pyr, dog_pyr;
	buildGaussianPyramid(_image, pyr, nOctaves);
	dog_pyr = buildDogPyr(pyr);
	getScaleSpaceExtrema(dog_pyr, keypoints);
	computeOrientationHist(dog_pyr, keypoints);
}

/**
 * Downsamples an image to quarter its size
 * half in each dimension
 *
 * @param image		The input image to downsample
 * @return Returns the resized image
 */
Mat downSample(Mat& image)
{
	Mat blurredImage;
	GaussianBlur(image, blurredImage, Size(0, 0), INTERPOLATION_SIGMA, 0);

	Mat temp = Mat(Size(blurredImage.cols / 2, blurredImage.rows), image.type());
	for (int i = 0; i < temp.cols; i++)
	{
		blurredImage.col(i * 2).copyTo(temp.col(i));
	}

	Mat resizedImage = Mat(Size(temp.cols, temp.rows / 2), image.type());
	for (int i = 0; i < resizedImage.rows; i++)
	{
		temp.row(i * 2).copyTo(resizedImage.row(i));
	}

	return resizedImage;
}

/**
 * Build Scale Space guassian pyramid from an image
 *
 * @param image		The base image of the pyramid
 * @param nOctaves	Number of Octaves
 * @param gauss_pyr	Guassian scale space pyramid
 *
 * @return gauss_pyr
 */
void buildGaussianPyramid(Mat& image, vector<vector<Mat> >& gauss_pyr, int nOctaves)
{
	double sigma;
	Mat tempImage;
	image.copyTo(tempImage);

	for (int i = 0; i < nOctaves; i++)
	{
		sigma = SIFT_INIT_SIGMA;
		vector<Mat> pyr_intervals;

		for (int j = 0; j < SIFT_INTVLS + 3; j++)
		{
			Mat blurredImage;
			GaussianBlur(tempImage, blurredImage, Size(0, 0), sigma, 0);
			pyr_intervals.push_back(blurredImage);
			sigma *= SIFT_STEP_SIGMA;
		}

		gauss_pyr.push_back(pyr_intervals);
		tempImage = downSample(tempImage);
	}
}

/**
 * Build difference of guassians Scale Space guassian
 * pyramid by subtracting every consecutive intervals
 *
 * @param gauss_pyr		Guassian scale space pyramid
 *
 * @return Returns the Difference of Guassians pyramid
 */
vector<vector<Mat> > buildDogPyr(vector<vector<Mat> > gauss_pyr)
{
	int nOctaves = gauss_pyr.size();
	vector<vector<Mat> > dog_pyr;

	for (int i = 0; i < nOctaves; i++)
	{
		vector<Mat> dog_intervals;

		for (int j = 0; j < SIFT_INTVLS + 2; j++)
		{
			dog_intervals.push_back(gauss_pyr[i][j] - gauss_pyr[i][j + 1]);
		}

		dog_pyr.push_back(dog_intervals);
	}

	return dog_pyr;
}

/**
 * Tests if the given point is an extrema by comparing
 * it to it's surroundings, bottom and top intervals
 *
 * @param dog_pyr		Difference of Guassians pyramid
 * @param octave		Current Octave
 * @param interval		Current Interval
 * @param r				Current row
 * @param c				Current column
 *
 * @return Returns true if extrema else false
 */
bool isExtrema(vector<vector<Mat> >& dog_pyr, int octave, int interval, int r, int c)
{
	float intensity = dog_pyr[octave][interval].at<float>(r, c);

	if (intensity > 0)
	{
		for (int i = -1; i <= 1; i++)
			for (int j = -1; j <= 1; j++)
				for (int k = -1; k <= 1; k++)
					if (intensity <= dog_pyr[octave][interval + i].at<float>(r + j, c + k) && (j != 0 || k != 0))
						return false;
	}
	else
	{
		for (int i = -1; i <= 1; i++)
			for (int j = -1; j <= 1; j++)
				for (int k = -1; k <= 1; k++)
					if (intensity >= dog_pyr[octave][interval + i].at<float>(r + j, c + k) && (j != 0 || k != 0))
						return false;
	}

	return true;
}

/**
 * Tests if the given point is an extrema by comparing
 * it to it's surroundings, bottom and top intervals
 *
 * @param dog_pyr		Difference of Guassians pyramid
 * @param keypoints		Keypoints vector
 *
 * @return list of keypoints
 */
void getScaleSpaceExtrema(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints)
{
	int octaves = dog_pyr.size();
	int intervals = dog_pyr[0].size() - 2;

	for (int i = 0; i < octaves; i++)
	{
		for (int j = 1; j <= intervals; j++)
		{
			for (int r = SIFT_IMG_BORDER; r < dog_pyr[i][0].rows - SIFT_IMG_BORDER; r++)
			{
				for (int c = SIFT_IMG_BORDER; c < dog_pyr[i][0].cols - SIFT_IMG_BORDER; c++)
				{
					if (isExtrema(dog_pyr, i, j, r, c))
						if (cleanPoints(Point(c, r), dog_pyr[i][j], SIFT_CURV_THR))
							keypoints.push_back(KeyPoint(c, r, j, -1, 0, i));
				}
			}
		}
	}
}

/**
 * Tests if the given point is a good feature or
 * not by discarding low contrast points and edges
 *
 * @param image			Current DOG scale space image
 * @param curv_thr		Curvature threshold
 *
 * @return true if good feature else false
 */
bool cleanPoints(Point position, Mat& image, int curv_thr)
{
	float rx, ry, fxx, fxy, fyy, deter;
	float trace, curvature;

	if (abs(image.at<float>(position)) < SIFT_CONTR_THR)
	{
		return false;
	}
	else
	{
		rx = position.x;
		ry = position.y;
		fxx = image.at<float>(rx - 1, ry) + image.at<float>(rx + 1, ry) - 2 * image.at<float>(rx, ry);
		fyy = image.at<float>(rx, ry - 1) + image.at<float>(rx, ry + 1) - 2 * image.at<float>(rx, ry);
		fxy = image.at<float>(rx - 1, ry - 1) + image.at<float>(rx + 1, ry + 1) - image.at<float>(rx - 1, ry + 1)
				- image.at<float>(rx + 1, ry - 1);

		trace = fxx + fyy;
		deter = (fxx * fyy) - (fxy * fxy);
		curvature = trace * trace / deter;

		if (deter < SIFT_DETER_THR || curvature > curv_thr)
		{
			return false;
		}
	}

	return true;
}

/**
 * Gets the first and second maximums
 * in a given histogram
 *
 * @param histogram		The given histogram
 * @param maximum		The first maximum value
 * @param indexMax		The index of first maximum
 * @param secondmax		The Second maximum value
 * @param indexSecond	The index of second maximum
 *
 * @return	maximum, indexMax, secondmax, indexSecond
 */
void histogramMax(vector<double> histogram, int &maximum, int &indexMax, int &secondmax, int &indexSecond)
{
	maximum = histogram[0];
	secondmax = histogram[0];
	indexMax = 0;
	indexSecond = 0;

	for (size_t i = 0; i < histogram.size(); i++)
	{
		if (maximum < histogram[i])
		{
			secondmax = maximum;
			indexSecond = indexMax;

			maximum = histogram[i];
			indexMax = i;
		}
	}
}

/**
 * Build a gradient histogram from
 * the given window and range
 *
 * @param matrix		Matrix Window
 * @param range			The histogram span
 * @param maximum		Maximum value in the histogram
 *
 * @return	Returns the calcuated histogram
 */
vector<double> buildHistogram(Mat matrix, int range, int maximum)
{
	int size = maximum / range;
	vector<double> histo(size);

	for (size_t i = 0; i < histo.size(); i++)
	{
		histo[i] = 0;
	}

	for (int i = 0; i < matrix.rows; i++)
	{
		for (int j = 0; j < matrix.cols; j++)
		{
			int index = matrix.at<float>(i, j) / range;
			histo[index]++;
		}
	}

	return histo;
}

/**
 * Compute the Orientation histogram
 * for the DOG pyramid and updates
 * the orientation of each keypoint
 *
 * @param dog_pyr		Difference of Guassians pyramid
 * @param keypoints		Keypoints vector
 *
 * @return	Returns keypointsGradients
 */
vector<Mat> computeOrientationHist(vector<vector<Mat> >& dog_pyr, vector<KeyPoint>& keypoints)
{
	int range = 10;
	int maximum = 360;
	vector<Mat> keypointsGradients;

	for (size_t z = 0; z < keypoints.size(); z++)
	{
		Mat image = dog_pyr[keypoints[z].octave][keypoints[z].size];
		int keyx = keypoints[z].pt.x;
		int keyy = keypoints[z].pt.y;

		if (keyx - SIFT_HIST_BOREDER - 1 < 0 || keyx + SIFT_HIST_BOREDER + 1 > image.cols
				|| keyy - SIFT_HIST_BOREDER - 1 < 0 || keyy + SIFT_HIST_BOREDER + 1 > image.rows)
		{
			return keypointsGradients;
		}
		else
		{
			Mat tempMagnitude = (Mat_<float>(SIFT_HIST_BOREDER * 2, SIFT_HIST_BOREDER * 2));
			Mat tempGradient = (Mat_<float>(SIFT_HIST_BOREDER * 2, SIFT_HIST_BOREDER * 2));
			for (int i = 0; i < SIFT_HIST_BOREDER * 2; i++)
			{
				for (int j = 0; j < SIFT_HIST_BOREDER * 2; j++)
				{
					float diffx, diffy, magnitude, gradient;

					diffx = image.at<float>(keyx + i + 1 - SIFT_HIST_BOREDER, keyy - SIFT_HIST_BOREDER + j)
							- image.at<float>(keyx + i - 1 - SIFT_HIST_BOREDER, keyy - SIFT_HIST_BOREDER + j);
					diffy = image.at<float>(keyx - SIFT_HIST_BOREDER + i, keyy + j + 1 - SIFT_HIST_BOREDER)
							- image.at<float>(keyx - SIFT_HIST_BOREDER + i, keyy + j - 1 - SIFT_HIST_BOREDER);
					magnitude = sqrt(pow(diffx, 2) + pow(diffy, 2));
					gradient = atan2f(diffy, diffx);

					if (gradient < 0)
					{
						gradient += (2 * PI);
					}
					gradient *= 360 / (2 * PI);

					tempMagnitude.at<float>(i, j) = magnitude;
					tempGradient.at<float>(i, j) = gradient;
				}
			}

			keypointsGradients.push_back(tempGradient);
			int maxima, secondmax, indexMax, indexSecond;

			vector<double> histo = buildHistogram(tempGradient, range, maximum);

			histogramMax(histo, maxima, indexMax, secondmax, indexSecond);

			int angleOrientation = indexMax * range;
			angleOrientation = angleOrientation + (range / 2);
			keypoints[z].angle = angleOrientation;
		}
	}

	return keypointsGradients;
}

/**
 * Compute the SIFT descriptor of
 * each keypoints
 *
 * @param keypointsGradients	Gradients Vector
 *
 * @return	Returns vector of all descriptors
 */
vector<vector<double> > computeDescriptors(vector<Mat> keypointsGradients)
{
	vector<vector<double> > descriptors;

	for (size_t points = 0; points < keypointsGradients.size(); points++)
	{
		vector<double> singleDescriptor;
		Mat temp = keypointsGradients[points];
		singleDescriptor.reserve(128);
		for (int xBlock = 0; xBlock < temp.cols; xBlock += 4)
		{
			for (int yBlock = 0; yBlock < 16; yBlock += 4)
			{
				Mat blockMatrix = Mat::zeros(4, 4, CV_32F);
				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						blockMatrix.at<float>(i, j) = temp.at<float>(xBlock + i, yBlock + j);
					}
				}

				vector<double> singleHistogram = buildHistogram(blockMatrix, 45, 360);
				singleDescriptor.insert(singleDescriptor.end(), singleHistogram.begin(), singleHistogram.end());
			}
		}

		descriptors.push_back(singleDescriptor);
	}

	return descriptors;
}

/**
 * Draws the given keypoints
 * on the given image
 *
 * @param keypoints		Keypoints vector
 * @param image			image to draw on
 *
 * @return	The updated image
 */
void drawKeyPoints(Mat& image, vector<KeyPoint>& keypoints)
{
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		Point pt1, pt2;
		pt1 = keypoints[i].pt;
		pt2.x = pt1.x + cos(keypoints[i].angle * PI / 180.0f) * pow(2, keypoints[i].octave);
		pt2.y = pt1.y + sin(keypoints[i].angle * PI / 180.0f) * pow(2, keypoints[i].octave);
		line(image, pt1 * pow(2, keypoints[i].octave), pt2 * pow(2, keypoints[i].octave), Scalar(150, 0, 0), 0.5,
		CV_AA);

		circle(image, pt1 * pow(2, keypoints[i].octave), 3, Scalar(80, 80, 80), -1, CV_AA);
		circle(image, pt1 * pow(2, keypoints[i].octave), 3, Scalar(0, 69, 255), -1, CV_AA);
	}
}
