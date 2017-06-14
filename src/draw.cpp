#include "draw.hpp"

using namespace cv;
using namespace std;
#define matched  1

void drawMatch(Mat img_1,
               vector<KeyPoint> keypoints1,
               vector<KeyPoint> keypoints2,
               vector<DMatch> matches,
               int radius,
               char* windowName){
    int ptsNum = (int)matches.size();
    float minDist = 10000;
    float dist;
    Point2f pt1, pt2;
    Mat copy;
    img_1.copyTo(copy);
    
    for(int n = 0; n < ptsNum; n++){
        dist = matches[n].distance;
        if(dist < minDist) minDist = dist;
        
    }
    
    for(int n = 0; n < ptsNum; n++){
        if(matches[n].distance <= 4*minDist){
            pt1 = keypoints1[matches[n].queryIdx].pt;
            pt2 = keypoints2[matches[n].trainIdx].pt;
            
            circle(copy, pt1, radius, Scalar(0,0,255));
            circle(copy, pt2, radius, Scalar(255,0,0));
            
            line(copy, pt1, pt2, Scalar(0,255,0));
        }
    }
    //-- Show detected matches
    imshow( windowName, copy );
    //    imshow("connected Features", copy);
    if(waitKey(1) == 27){
        exit;
    }
    
}

void drawMatch(Mat img_1,
               vector<KeyPoint> keypoints1,
               vector<KeyPoint> keypoints2,
               int radius,
               char* windowName){
    int ptsNum = (int)keypoints1.size();
    Point2f pt1, pt2;
    Mat copy;
    img_1.copyTo(copy);
    for(int n = 0; n < ptsNum; n++){
        pt1 = keypoints1[n].pt;
        pt2 = keypoints2[n].pt;
        
        circle(copy, pt1, radius, Scalar(0,0,255));
        circle(copy, pt2, radius, Scalar(255,0,0));
        
        line(copy, pt1, pt2, Scalar(0,255,0));
    }
    imshow(windowName, copy);
    if(waitKey(1) == 27){
        exit;
    }
}

void drawMatch(Mat img_1,
               vector<Point2f> p_keypoints1,
               vector<Point2f> p_keypoints2,
               int radius,
               char* windowName){
    int ptsNum = (int)p_keypoints1.size();
    Point2f pt1, pt2;
    Mat copy;
    img_1.copyTo(copy);
    for(int n = 0; n < ptsNum; n++){
        pt1 = p_keypoints1[n];
        pt2 = p_keypoints2[n];
        
        circle(copy, pt1, radius, Scalar(0,0,255));
        circle(copy, pt2, radius, Scalar(255,0,0));
        
        line(copy, pt1, pt2, Scalar(0,255,0));
    }
    imshow(windowName, copy);
    if(waitKey(1) == 27){
        exit;
    }
}



void drawFeature(Mat img, vector<KeyPoint> features, char* windowName){
    
    int ptsNum = (int)features.size();
    vector<Point2f> p_features;
    p_features.resize(ptsNum);
    
    Mat copy;
    copy = img.clone();
    KeyPoint::convert(features, p_features);
    for( int i = 0; i < p_features.size(); i++ ){
        circle( copy, p_features[i], 2, Scalar(0,250,0), 1, 8, 0 );
    }
    //    cout << "points number: "<< p_features.size() << "\n";
    imshow(windowName, copy);
    if(waitKey(1) == 27){
        exit;
    }
    
}


void drawFeature(Mat img, vector<Point2f> features, char* windowName){
    Mat copy;
    copy = img.clone();
    for( int i = 0; i < features.size(); i++ ){
        //        circle( copy, features[i], 4, Scalar(rng.uniform(0,255), rng.uniform(0,255),rng.uniform(0,255)), -1, 8, 0 );
        circle( copy, features[i], 2, Scalar(0, 255,0), 0.5, 1, 0 );
    }
    //    cout << "points number: "<< features.size() << "\n";
    imshow(windowName, copy);
    if(waitKey(1) == 27){
        exit;
    }
}

void drawFarandCloseFeatures(Mat img, vector<Point2f> closePts, 
                             vector<Point2f> farPts, const char* windowName){
    Mat copy;
    copy = img.clone();
    // cout << "draw far ponits: " << farPts.size() << endl;
    // cout << "draw close points: " << closePts.size() << endl;
    for( int i = 0; i < closePts.size(); i++){
    // for( int i = 0; i < 20; i++){
        circle( copy, closePts[i], 2, Scalar(0, 255,0), 0.5, 1, 0 );
    }
    for( int i = 0; i < farPts.size(); i++){
        circle( copy, farPts[i], 4, Scalar(0, 0, 255), 0.5, 1, 0 );
    }
    imshow(windowName, copy);
    if(waitKey(1) == 27){
       exit;
    }
}

void drawFarandCloseFeatures(Mat img, vector<Point2f> pts, 
                             vector<int> farIdx, char* windowName){

    Mat copy;
    copy = img.clone();
    for( int i = 0; i < farIdx.size(); i++){
        if(farIdx[i] == 0)
            circle( copy, pts[i], 2, Scalar(0, 255,0), 0.5, 1, 0 );
        else
            circle( copy, pts[i], 2, Scalar(0, 0, 255), 0.5, 1, 0 );
    }
    imshow(windowName, copy);
    if(waitKey(1) == 27){
       exit;
    }


}
