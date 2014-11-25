#include "SIFT.h"

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
		blurredImage.col(i * 2).copyTo(temp.col(i));

	Mat resizedImage = Mat(Size(temp.cols, temp.rows / 2), image.type());
	for (int i = 0; i < resizedImage.rows; i++)
		temp.row(i * 2).copyTo(resizedImage.row(i));

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
			blurredImage.release();
			sigma *= SIFT_STEP_SIGMA;
		}

		gauss_pyr.push_back(pyr_intervals);
		tempImage = downSample(tempImage);
	}

	tempImage.release();
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
bool cleanPoints(Point loc, Mat& image, int curv_thr)
{
	float rx, ry, fxx, fxy, fyy, deter;
	float trace, curvature;

	if (abs(image.at<float>(loc)) < SIFT_CONTR_THR)
		return false;
	else
	{
		rx = loc.x;
		ry = loc.y;
		fxx = image.at<float>(rx - 1, ry) + image.at<float>(rx + 1, ry) - 2 * image.at<float>(rx, ry);
		fyy = image.at<float>(rx, ry - 1) + image.at<float>(rx, ry + 1) - 2 * image.at<float>(rx, ry);
		fxy = image.at<float>(rx - 1, ry - 1) + image.at<float>(rx + 1, ry + 1) - image.at<float>(rx - 1, ry + 1)
				- image.at<float>(rx + 1, ry - 1);

		trace = fxx + fyy;
		deter = (fxx * fyy) - (fxy * fxy);
		curvature = trace * trace / deter;

		if (deter < SIFT_DETER_THR || curvature > curv_thr)
			return false;
	}

	return true;
}
