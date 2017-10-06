#ifndef SYSTEM_H
#define SYSTEM_H

#include <stdio.h>

#include "utils.h"
#include "Frame.h"
#include "Map.h"
#include "Mapviewer.h"
#include "optimizer.hpp"
#include "PoseOpt.h"
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>

void SLAMsystem(std::string commonPath, std::string yamlPath);

#endif
