#ifndef FRAME_H
#define FRAME_H

#include "MapPoint.h"
#include "Map.h"
#include "utils.h"
#include "ORBextractor.h"

#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <limits>

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/features2d/features2d.hpp>

typedef std::unordered_map<unsigned int, std::pair<cv::Mat, cv::Mat> > edgeConstrain;

class Map;
class MapPoint;
class Frame{
public:
	unsigned int frameID;
	STEREO_RECTIFY_PARAMS srp;
	cv::Mat K;
	double fx;
	double fy;
	double cx;
	double cy;
	double b;

	cv::Mat imgL, imgR;
	cv::Mat despL, despR;

	cv::Mat rvec, tvec;
	cv::Mat worldRvec, worldTvec;

	std::vector<cv::KeyPoint> keypointL, keypointR;
	std::vector<cv::Point3f> scenePts;        //scene points in the current frame coordinate
	std::vector<cv::Point3f> scenePtsinWorld; //scene points in the world corrdinate
	
	std::vector<cv::DMatch> matchesBetweenFrame;

	edgeConstrain relativePose; //record relative transformation
	std::vector<MapPoint*> mappoints; // 3d points in world coordinate
	std::vector<bool> originality;
	Map* map;

public:
	Frame(string leftImgFile, string rightImgFile,
		  STEREO_RECTIFY_PARAMS _srp, int _id, Map* _map);

	void matchFrame(Frame* frame, bool useMappoints = false);
	void addEdgeConstrain(unsigned int id, cv::Mat relativeRvec, cv::Mat relativeTvec);
	void manageMapPoints(Frame* frame);

	void setWrdTransVectorAndTransScenePts(cv::Mat _worldRvec, cv::Mat _worldTvec);
	void transformScenePtsToWorldCoordinate();
    
    void matchFeatureKNN(const cv::Mat& desp1, const cv::Mat& desp2, 
                            const std::vector<cv::KeyPoint>& keypoint1, 
                            const std::vector<cv::KeyPoint>& keypoint2,
                            std::vector<cv::KeyPoint>& matchedKeypoint1,
                            std::vector<cv::KeyPoint>& matchedKeypoint2,
                            std::vector<cv::DMatch>& matches,
                            double knn_match_ratio = 0.8,
                            bool hamming = false);

    void compute3Dpoints(std::vector<cv::KeyPoint>& kl, 
					 	 std::vector<cv::KeyPoint>& kr,
					 	 std::vector<cv::KeyPoint>& trikl,
					 	 std::vector<cv::KeyPoint>& trikr,
					 	 std::vector<int>& inliers);

	void PnP(std::vector<cv::Point3f> obj_pts, 
	                std::vector<cv::Point2f> img_pts,
	                cv::Mat& inliers);
    void judgeBadPoints();
    void judgeBadPointsKdTree();
	MapPoint* createNewMapPoint(unsigned int pointIdx);
	void pointToExistingMapPoint(Frame* frame, MapPoint* mp, unsigned int currIdx);
	Eigen::Affine3d getWorldTransformationMatrix();


    void releaseMemory();
};
	

#endif