#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "Frame.h"
#include <set>


class MapPoint;
class Frame;

class Map{
public:
	Map();
	void addMapPoint(MapPoint* mappoint);
	std::vector<cv::Point3f> getAllMapPoints();
	long unsigned int allMapPointNumber();
	void getMapPointsFromFrame(Frame* frame);

	std::set<MapPoint*> allMapPoints;
private:
	long unsigned int pointIdx; 

};


#endif