#include "SIFT.h"

static void help()
{
	cout << "\nThis program illustrates the use of SIFT detector and descriptor\n";
	cout << "Provide the path to the target image as an argument.\n";
	cout << "Call:\n"
			"    /.SIFT [image_name]\n";
	cout << "\nHot keys: \n"
			"\tESC - quit the program\n";
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		help();
		return 1;
	}

	string filename = argv[1];
	if (filename.empty())
	{
		cout << "\nDurn, couldn't read in " << argv[1] << endl;
		return 1;
	}
	Mat image = imread(filename, 1);

	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}

	help();

	SIFT detector;
	vector<KeyPoint> keypoints;
	detector.findSiftInterestPoint(image, keypoints);
	detector.drawKeyPoints(image, keypoints);
	imshow("SIFT features", image);

	waitKey(0);
	return 0;
}
