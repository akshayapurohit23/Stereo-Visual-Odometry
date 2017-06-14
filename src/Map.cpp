#include "Map.h"

Map::Map():pointIdx(0){
}

void Map::addMapPoint(MapPoint* mappoint){
	allMapPoints.insert(mappoint);
	pointIdx++;
}

std::vector<cv::Point3f> Map::getAllMapPoints(){
	std::vector<cv::Point3f> result;

	for(auto mp : allMapPoints){
		if(!mp->isBad){ result.push_back(mp->pos); }
	}

	return result;
}


long unsigned int Map::allMapPointNumber(){
	return (long unsigned int)allMapPoints.size();
}




