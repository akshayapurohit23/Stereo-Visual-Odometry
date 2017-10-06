#ifndef MAPVIEWER_H
#define MAPVIEWER_H

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/core/core.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

class Mapviewer{

public:
	std::vector<pcl::PointCloud<pcl::PointXYZRGB> > allCameras;
	pcl::PointCloud<pcl::PointXYZRGB> templateCamera;


	pcl::PointCloud<pcl::PointXYZRGB> entireMap;// (new PointCloud<PointXYZ>);
	bool initialized = false;

	Mapviewer();
	void jointToMap(pcl::PointCloud<pcl::PointXYZRGB> frameMap, Eigen::Affine3d& trans);
	pcl::PointCloud<pcl::PointXYZRGB> pointToPointCloud(std::vector<cv::Point3f> scenePts,
													 int R=255, int G=255, int B=255);
	
	void addMorePoints(pcl::PointCloud<pcl::PointXYZRGB> frameMap, Eigen::Affine3d& trans, bool downsample = false);
	void addCamera(Eigen::Affine3d& trans);
};

#endif
