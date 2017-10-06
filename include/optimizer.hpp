#ifndef optimizer_hpp
#define optimizer_hpp

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <fstream>
#include <algorithm>
#include <unordered_set>

//ceres
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include <ceres/local_parameterization.h>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/types.h>
#include <ceres/rotation.h>
#include "snavely_reprojection_error.h"
#include "bal_problem.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include "Frame.h"
#include "Map.h"
#include "MapPoint.h"
#include "draw.hpp"
#include "utils.h"

using namespace std;
using namespace ceres;
using namespace examples;

class Frame;
class Map;
class MapPoint;

class Optimizer{
private:
    Solver::Options options;
    Solver::Summary summary;
    const int pointBlkSize  = 3;
    const int cameraBlkSize = 6;

public:
    Problem& globalBAProblem;
	Optimizer(bool printOut, Problem& prob);

    void globalBundleAdjustment(Map* map, vector<Frame*> frames);
    void localBundleAdjustment(vector<Frame*> frames, int startIdx, int length);
	void poseOptimization(vector<Frame* >& frames);
    void reprojectionOnlyAdjustment(Map* map, vector<Frame*> frames);
	void solveProblem(Problem& pb);
};


#endif
