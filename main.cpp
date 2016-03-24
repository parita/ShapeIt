#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>
#include "HandDetector.h"
#include "HandDetector.cpp"
#include <iostream>

using namespace cv;
using namespace std;

cv::Point sphere_centre;
cv::Point radius;	

int main()
{
    VideoCapture cap(0);
    if( !cap.isOpened() )
    {

        puts("***Could not initialize capturing...***\n");
        return 0;
    }
    Mat frame,foreground,image;
    BackgroundSubtractorMOG2 mog;
    int fps=cap.get(CV_CAP_PROP_FPS);
    if(fps<=0)
        fps=10;
    else
        fps=1000/fps;

	HandDetector::Params p;
	p.area=1000;		//minimum area for qualifying a contour
	p.cosThreshold=0.5;	//Threshold cosine value for detecting the angle between two vectors in the contour for fingertip detection
	p.equalThreshold=1e-7;	//Threshold for two values within which the two values are still considered equal
	p.r=40;
	p.step=16;

	HandDetector hDetector;
	hDetector.setParams(p);
	std::vector<Hand> hands;

    int frame_no=0, foreground_no=0, mask_no=0, depth_no=0;
    for(;;)
    {
        cap>>frame;   
        if( frame.empty() )
                break;
        image=frame.clone();
        mog(frame,foreground,-1);
        	
	Mat gray_image;
	cvtColor(image, gray_image, CV_RGB2GRAY);
	
        threshold(foreground,foreground,128,255,THRESH_BINARY);
        medianBlur(foreground,foreground,9);
        erode(foreground,foreground,Mat());
        dilate(foreground,foreground,Mat());

	Mat crop;
	crop = foreground & gray_image;
	
    Mat circle_mask = Mat::zeros(crop.size(), CV_8UC1);
    
    vector<Vec3f> circles;
    HoughCircles(crop, circles, CV_HOUGH_GRADIENT, 1, 10, 100, 30, 1, 100);		//Detects the circles in the image

//---Calculate the circle with the maximum radius in the given radius range----//
    int max_radius=0;
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
	
	if(c[2]>max_radius)
	{
		sphere_centre = Point(c[0],c[1]);
		max_radius = c[2];
	}	
    }
		
	if(max_radius>0)
	{
	circle_mask = Mat::zeros(crop.size(), CV_8UC1);
	circle( circle_mask, sphere_centre,max_radius, 255, max_radius+100, CV_AA);	//Creates a mask to hide the circle detected when processing for fingertip detection
	}
	else;
   
    cout<<"size:"<<circles.size()<<endl;
    bitwise_not(circle_mask, circle_mask);	//invert the mask before it can be ANDed with the forground

	cv::Mat depthMap;
	cv::Mat grayImage;
	cv::Mat tmp;
	depthMap = crop & circle_mask;
	Mat cimg;
		cv::cvtColor(depthMap, tmp, CV_GRAY2BGR);

		cv::threshold(depthMap, depthMap, 60, 255, cv::THRESH_BINARY);
	    if(circles.size())
	    {

		hDetector.detect(depthMap, hands);
		if(!hands.empty())
		{
			drawHands(depthMap, hands);
			radius = hDetector.getMaxDist(hands);	//Function that returns the fingertip point co-ordinates

		int dist_ = dist(sphere_centre,radius);		//Calculates the distance between the sphere centre and the finger-tip

		if(dist_>0)
		{
		circle( image, sphere_centre, dist_, Scalar(0,255,0), 3, CV_AA);		//Draw the modified circle
		circle( image, radius, 2, Scalar(0,0,255), 3, CV_AA);				//Draw the finger-tip point
		circle( image, sphere_centre, 2, Scalar(255,0,0), 3, CV_AA);			//Draw the detected sphere's centre
		circle( image, sphere_centre, max_radius, Scalar(255,0,0), 3, CV_AA);		//Draw the original sphere
		
		// Saving results
		std::stringstream out_name;	//To define a name for the processed frames to be saved
		out_name << "test" << frame_no << ".jpg";
		imwrite(out_name.str(), image);	//Write the frame to .jpg file
		frame_no++;

		//Saving the foreground for fingertip detection		
		std::stringstream depth_name;	//To define a name for the processed frames to be saved
		depth_name << "depth" << depth_no << ".jpg";
		imwrite(depth_name.str(), depthMap);	//Write the frame to .jpg file
		depth_no++;

		//Saving the mask created by the detected circles				
		std::stringstream mask_name;	//To define a name for the processed frames to be saved
		mask_name << "mask" << mask_no << ".jpg";
		imwrite(mask_name.str(), circle_mask);	//Write the frame to .jpg file
		mask_no++;		
		
		//Saving initial foreground
		std::stringstream fore_name;	//To define a name for the processed frames to be saved
		fore_name << "fore" << foreground_no << ".jpg";
		imwrite(fore_name.str(), foreground);	//Write the frame to .jpg file
		foreground_no++;
		}
		
		}

	imshow("Masked foreground", depthMap);
	}
	imshow("mask",circle_mask);
	imshow("output", image);
	
        char c = (char)waitKey(fps);
        if( c == 27 )   // Exits when ESC is pressed
            break;

    }


}
