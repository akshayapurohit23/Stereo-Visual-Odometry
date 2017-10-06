#include "utils.h"
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace pcl;

vector<string> getImgFileName(string &strPathImg){
    string cpstrPathImg = strPathImg;
    vector<string> vstrImage;

    DIR *dir;
    class dirent *ent;
    class stat st;
    dir = opendir(cpstrPathImg.c_str());
    while ((ent = readdir(dir)) != NULL) {
        const string file_name = ent->d_name;
        const string full_file_name = cpstrPathImg + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if (is_directory)
            continue;

        vstrImage.push_back(full_file_name);
    }
    closedir(dir);
    sort(vstrImage.begin(), vstrImage.end());
    return vstrImage;
}

void transformationToVector(Eigen::Affine3d transformMatrix, Mat& rvec, Mat& tvec)
{
    Mat R = Mat::zeros(3,3,CV_64F)  ;
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<double>(r,c) = transformMatrix(r,c);
        }
    }
    Rodrigues(R, rvec);
    tvec = Mat::zeros(3,1,CV_64F);
    tvec.at<double>(0,0) = transformMatrix(0,3);
    tvec.at<double>(1,0) = transformMatrix(1,3);
    tvec.at<double>(2,0) = transformMatrix(2,3);
}

void convertTransformMatrixToVector(vector<Eigen::Affine3d> TransformaMatrix,
                               	    vector<Mat>& Rvec,
                                    vector<Mat>& Tvec){

    int numTrans = TransformaMatrix.size();
    Rvec.clear();
    Tvec.clear();

    for(int n = 0; n < numTrans; n++){
        Mat rvec;
        Mat tvec;
        transformationToVector(TransformaMatrix[n], rvec, tvec);
        Rvec.push_back(rvec);
        Tvec.push_back(tvec);
    }
}

void getAccumulateMotion(const vector<Mat>& Rvec,
                         const vector<Mat>& Tvec,
                         int startIdx, int endIdx,
                         Mat& accumRvec, Mat& accumTvec){
    accumRvec = Mat::zeros(3,1,CV_64F);
    accumTvec = Mat::zeros(3,1,CV_64F);
    Eigen::Affine3d accumMatrix = vectorToTransformation(accumRvec, accumTvec);
    for(int n = startIdx; n < endIdx; n++){
        Mat tempRvec, tempTvec;
        tempRvec = Rvec[n].clone();
        tempTvec = Tvec[n].clone();
        Eigen::Affine3d tempMatrix = vectorToTransformation(tempRvec, tempTvec);
        accumMatrix = tempMatrix * accumMatrix;
    }

    Mat R = Mat::zeros(3,3,CV_64F);
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<double>(r,c) = accumMatrix(r,c);
        }
    }

    Rodrigues(R, accumRvec);
    accumTvec.at<double>(0,0) = accumMatrix(0,3);
    accumTvec.at<double>(1,0) = accumMatrix(1,3);
    accumTvec.at<double>(2,0) = accumMatrix(2,3);
}

void getRelativeMotion(const Mat startPosR, const Mat startPosT,
                       const Mat endPosR, const Mat endPosT,
                       Mat& r2to1, Mat& t2to1){
    r2to1 = Mat::zeros(3,1,CV_64F);
    t2to1 = Mat::zeros(3,1,CV_64F);

    Eigen::Affine3d startTrans, endTrans, trans1to2;
    Mat startPosRMat, endPosRMat;
    Rodrigues(startPosR, startPosRMat);
    Rodrigues(endPosR, endPosRMat);
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            startTrans(r,c) = startPosRMat.at<double>(r,c);
            endTrans(r,c) = endPosRMat.at<double>(r,c);
        }
    }

    startTrans(0,3) = startPosT.at<double>(0,0);
    startTrans(1,3) = startPosT.at<double>(1,0);
    startTrans(2,3) = startPosT.at<double>(2,0);
    endTrans(0,3) = endPosT.at<double>(0,0);
    endTrans(1,3) = endPosT.at<double>(1,0);
    endTrans(2,3) = endPosT.at<double>(2,0);

    trans1to2 = endTrans * startTrans.inverse();
    Mat R = Mat::zeros(3,3,CV_64F);
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<double>(r,c) = trans1to2(r,c);
        }
    }
    Rodrigues(R, r2to1);

    t2to1.at<double>(0,0) = trans1to2(0,3);
    t2to1.at<double>(1,0) = trans1to2(1,3);
    t2to1.at<double>(2,0) = trans1to2(2,3);
}

Eigen::Affine3d vectorToTransformation(Mat rvec, Mat tvec){
	Mat R;
	Eigen::Affine3d result;
	Rodrigues(rvec, R);
	for(int r = 0; r < 3; r++){
		for(int c = 0; c < 3; c++){
			result(r,c) = R.at<double>(r,c);
		}
	}
	result(0, 3) = tvec.at<double>(0, 0);
	result(1, 3) = tvec.at<double>(1, 0);
	result(2, 3) = tvec.at<double>(2, 0);
	return result;
}

vector<Point3f> transformPoints(Eigen::Affine3d trans, vector<Point3f> obj_pts){
	vector<Point3f> result;
	PointCloud<PointXYZ> tempCloud = Point3ftoPointCloud(obj_pts);
	Eigen::Affine3d invTrans = trans.inverse();
	transformPointCloud(tempCloud, tempCloud, invTrans.matrix());
	result = PointCloudtoPoint3f(tempCloud);
	return result;
}


Point3f transformPoint(Eigen::Affine3d trans, Point3f pt){
    Point3f result;
    Eigen::Affine3d invTrans = trans.inverse();
    result.x = (float)invTrans(0,0)*pt.x + (float)invTrans(0,1)*pt.y + (float)invTrans(0,2)*pt.z + (float)invTrans(0,3);
    result.y = (float)invTrans(1,0)*pt.x + (float)invTrans(1,1)*pt.y + (float)invTrans(1,2)*pt.z + (float)invTrans(1,3);
    result.z = (float)invTrans(2,0)*pt.x + (float)invTrans(2,1)*pt.y + (float)invTrans(2,2)*pt.z + (float)invTrans(2,3);
    return result;
}

vector<Point3f> PointCloudtoPoint3f(PointCloud<PointXYZ> ptcloud){
    PointXYZ ptxyz;
    Point3f pt;
    vector<Point3f> result;
    for(int i = 0; i < ptcloud.points.size(); i++){
        ptxyz = ptcloud.points[i];
        pt.x = ptxyz.x;
        pt.y = ptxyz.y;
        pt.z = ptxyz.z;

        result.push_back(pt);
    }
    return result;
}

vector<Point3f> PointCloudtoPoint3f(PointCloud<PointXYZI> ptcloud){
    PointXYZI ptxyzi;
    Point3f pt;
    vector<Point3f> result;
    for(int i = 0; i < ptcloud.points.size(); i++){
        ptxyzi = ptcloud.points[i];
        pt.x = ptxyzi.x;
        pt.y = ptxyzi.y;
        pt.z = ptxyzi.z;
        result.push_back(pt);
    }
    return result;
}

PointCloud<PointXYZ> Point3ftoPointCloud(vector<Point3f> pts){
    PointXYZ ptxyz;
    Point3f pt;
    PointCloud<PointXYZ> result;
    for(int i = 0; i < pts.size(); i++){
        pt = pts[i];
        ptxyz.x = pt.x;
        ptxyz.y = pt.y;
        ptxyz.z = pt.z;
        result.points.push_back(ptxyz);
    }
    return result;
}

double normOfTransform( cv::Mat rvec, cv::Mat tvec ){
    return fabs(MIN(norm(rvec), 2*M_PI-norm(rvec)))+ fabs(norm(tvec));
}

Point3f getCameraCenter(const Mat Rvec,
                        const Mat Tvec){
    Point3f cameraCenter;
    Mat worldCenter = Mat::zeros(4,1,CV_64F);
    Mat homCameraCenter = Mat::zeros(4,1,CV_64F);
    worldCenter.at<double>(3,0)=1;
    Eigen::Affine3d accumMatrix = vectorToTransformation(Rvec, Tvec);
    for (int i=0;i<4;i++){
	double sum=0;
	for (int j=0;j<4;j++){
	    sum+=accumMatrix(i,j)*worldCenter.at<double>(j);
	}
	homCameraCenter.at<double>(i,0)=sum;
    }
    cameraCenter.x = homCameraCenter.at<double>(0,0)/homCameraCenter.at<double>(3,0);
    cameraCenter.y = homCameraCenter.at<double>(1,0)/homCameraCenter.at<double>(3,0);
    cameraCenter.z = homCameraCenter.at<double>(2,0)/homCameraCenter.at<double>(3,0);
    return cameraCenter;
}
