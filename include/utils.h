#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <fstream>
#include <algorithm>    // std::sort
#include <sys/types.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "draw.hpp"

using namespace std;
struct STEREO_RECTIFY_PARAMS{
    cv::Mat P1, P2;
    cv::Mat R1, R2;
    cv::Mat Q;
    cv::Mat map11, map12;
    cv::Mat map21, map22;
    cv::Size imageSize;
};

std::vector<string> getIMUFileName(string &strPathIMU);

std::vector<string> getPCDFileName(string &strPathPCD);

std::vector<string> getImgFileName(string &strPathImg);

std::vector<Eigen::Affine3d> loadGPSIMU(const std::vector<string>& strPathIMU);

void getRectificationParams(cv::Mat K1, cv::Mat K2, 
                            cv::Mat R, cv::Mat t,
                            cv::Mat distort1,
                            cv::Mat distort2,
                            STEREO_RECTIFY_PARAMS& srp);

std::vector<Eigen::Affine3d> getTransformFromGPSIMUinCamera(std::vector<std::string> IMUFileName);

void GetTransformationFromEulerAngleAndTranslation(Eigen::Affine3d& Transformation,
                                                   double rx, double ry, double rz, 
                                                   double tx, double ty, double tz);


void convertTransformMatrixToVector(std::vector<Eigen::Affine3d> TransformaMatrix,
                               		  std::vector<cv::Mat>& Rvec,
                               		  std::vector<cv::Mat>& Tvec);

void convertGPScoordinateToTranslation(double lati, double lon, double alt,
                                       double& tx, double& ty, double& tz);

void GetTransformationFromRollPitchYawAndTranslation(Eigen::Affine3d& Transformation,
                                                     double roll, double pitch, double yaw, 
                                                     double tx, double ty, double tz);

void getAccumulateMotion(const std::vector<cv::Mat>& Rvec,
                         const std::vector<cv::Mat>& Tvec,
                         int startIdx, int endIdx,
                         cv::Mat& accumRvec, cv::Mat& accumTvec);

void getRelativeMotion(const cv::Mat startPosR, const cv::Mat startPosT,
						           const cv::Mat endPosR, const cv::Mat endPosT,
					             cv::Mat& r2to1, cv::Mat& t2to1);

Eigen::Affine3d vectorToTransformation(cv::Mat rvec, cv::Mat tvec);

void transformationToVector(Eigen::Affine3d trans, cv::Mat& rvec, cv::Mat& tvec);

std::vector<cv::Point3f> transformPoints(Eigen::Affine3d trans, std::vector<cv::Point3f> obj_pts);

cv::Point3f transformPoint(Eigen::Affine3d trans, cv::Point3f pt);

pcl::PointCloud<pcl::PointXYZ> Point3ftoPointCloud(std::vector<cv::Point3f> pts);

std::vector<cv::Point3f> PointCloudtoPoint3f(pcl::PointCloud<pcl::PointXYZ> ptcloud);

std::vector<cv::Point3f> PointCloudtoPoint3f(pcl::PointCloud<pcl::PointXYZI> ptcloud);

std::vector<cv::Point3f> readLiDARandConvertToWorldCoor(string PCDPath, 
                                               cv::Mat accumRvec, 
                                               cv::Mat accumTvec);
double normOfTransform( cv::Mat rvec, cv::Mat tvec );
cv::Point3f getCameraCenter(const cv::Mat Rvec,
                            const cv::Mat Tvec);
#endif
