#ifndef __HAND_DETECTOR_H__
#define __HAND_DETECTOR_H__

#include <iostream>
#include <vector>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/*
#include <core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
*/
class Hand
{
public:
	Hand(){};
	~Hand(){};
	std::vector<cv::Point> fingers;
	cv::Point center;
	std::vector<cv::Point> contour;
};

class HandDetector
{
public:
	struct Params
	{
		int area;
		int r;
		int step;
		double cosThreshold;
		double equalThreshold;
	};

public:
	HandDetector(){};
	~HandDetector(){};

	void detect(cv::Mat& mask, std::vector<Hand>& hands);
	void setParams(Params& p);
	cv::Point getMaxDist(std::vector<Hand>& hands);
private:
	Params param;
	signed int rotation(std::vector<cv::Point>& contour, int pt, int r);
	double angle(std::vector<cv::Point>& contour, int pt, int r);
	bool isEqual(double a, double b);
};
double dist(cv::Point p1, cv::Point p2);
void drawHands(cv::Mat& image, std::vector<Hand>& hands);

#endif
