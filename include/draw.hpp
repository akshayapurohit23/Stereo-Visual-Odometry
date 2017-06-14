#ifndef draw_hpp
#define draw_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/calib3d/calib3d.hpp"



void drawFeature(cv::Mat img, std::vector<cv::Point2f> features, char* windowName);
void drawFeature(cv::Mat img, std::vector<cv::KeyPoint> features, char* windowName);

void drawFarandCloseFeatures(cv::Mat img, std::vector<cv::Point2f> closePts, 
                             std::vector<cv::Point2f> farPts, const char* windowName);

void drawFarandCloseFeatures(cv::Mat img, std::vector<cv::Point2f> pts, 
                             std::vector<int> farIdx, char* windowName);
void drawMatch(cv::Mat img_1,
               std::vector<cv::KeyPoint> keypoints1,
               std::vector<cv::KeyPoint> keypoints2,
               std::vector<cv::DMatch> matches,
               int radius,
               char* windowName);

void drawMatch(cv::Mat img_1,
               std::vector<cv::KeyPoint> keypoints1,
               std::vector<cv::KeyPoint> keypoints2,
               int radius,
               char* windowName);

void drawMatch(cv::Mat img_1,
               std::vector<cv::Point2f> p_keypoints1,
               std::vector<cv::Point2f> p_keypoints2,
               int radius,
               char* windowName);

#endif /* draw_hpp */
