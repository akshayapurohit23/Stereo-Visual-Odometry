#ifndef PoseOpt_h
#define PoseOpt_h

#include <stdio.h>
#include <iostream>
// #include <iomanip>
#include <time.h>
#include <string.h>
// #include <dirent.h>
#include <fstream>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include "utils.h"
#include "Frame.h"
#include "MapPoint.h"

#include <Eigen/Core>
#include <Eigen/Geometry>


typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 


class PoseOpt {
public:
	g2o::SparseOptimizer optimizer;
	g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Welsch" );

public:
	PoseOpt();
	void addNode(const std::vector<Frame*>& frames, int idx);
	void addEdge(const std::vector<Frame*>& frames, int fromIdx);
	
	void solve();	

};






#endif