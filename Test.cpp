#include "SIFT.h"

int main(int argc, char** argv)
{
	Mat imageColor = imread("../Test-Data/images/Lenna.png", CV_LOAD_IMAGE_COLOR);

	if (!imageColor.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	vector<KeyPoint> keypoints;
	SIFT detector;
	detector.findSiftInterestPoint(imageColor, keypoints);
	detector.drawKeyPoints(imageColor, keypoints);
	imshow("SIFT features", imageColor);

	waitKey(0);
	return 0;
}
