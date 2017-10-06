#include "Mapviewer.h"

using namespace pcl;
using namespace std;

Mapviewer::Mapviewer(){
	float cameraSize = 0.1;
	for(float x = -cameraSize; x < cameraSize; x+=(cameraSize/2)){
		for(float y = -cameraSize; y < cameraSize; y+=(cameraSize/2)){
			for(float z = -cameraSize; z < cameraSize; z+=(cameraSize/2)){
				PointXYZRGB pt;
				pt.x = x;
				pt.y = y;
				pt.z = z;
				pt.r = 255;
				pt.g = 0;
				pt.b = 0;
				templateCamera.push_back(pt);
			}}}
}

pcl::PointCloud<pcl::PointXYZRGB> Mapviewer::pointToPointCloud(std::vector<cv::Point3f> scenePts, 
														    int R, int G, int B){
	int ptsNum = scenePts.size();
	PointCloud<PointXYZRGB> result;
	for(int n = 0; n < ptsNum; n++){
		PointXYZRGB pt;
		pt.x = scenePts[n].x;
		pt.y = scenePts[n].y;
		pt.z = scenePts[n].z;
		pt.r = R;
		pt.g = G;
		pt.b = B;
		result.points.push_back(pt);
	}

	result.height = 1;
	result.width = result.points.size();
	return result;
}


void Mapviewer::jointToMap(PointCloud<PointXYZRGB> frameMap, Eigen::Affine3d& trans){
	if(false == initialized){
		initialized = true;
		cout << "map initializing! " << endl;
		entireMap = frameMap;
		cout << "map initialized!" << endl;
	}
	entireMap = frameMap;
	for(auto cam : allCameras)
		entireMap += cam;
}

void Mapviewer::addMorePoints(PointCloud<PointXYZRGB> frameMap, Eigen::Affine3d& trans, bool downsample){
	if(false == initialized){
		initialized = true;
		cout << "map initializing! " << endl;
		entireMap = frameMap;
		cout << "map initialized!" << endl;
	}
	if(downsample == true){
		PointCloud<PointXYZRGB>::Ptr cloud  (new PointCloud<PointXYZRGB>);
		*cloud = frameMap;
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.3f, 0.3f, 0.3f);
        sor.filter (*cloud);
        frameMap = *cloud;
	}
	entireMap += frameMap;
}

void Mapviewer::addCamera(Eigen::Affine3d& trans){
	//generata a set of points
	Eigen::Affine3d invTrans = trans.inverse();
	PointCloud<PointXYZRGB> curCamera;
	transformPointCloud(templateCamera, curCamera, invTrans);
	allCameras.push_back(curCamera);
}
