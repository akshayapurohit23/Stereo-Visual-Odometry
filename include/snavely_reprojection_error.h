// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Templated struct implementing the camera model and residual
// computation for bundle adjustment used by Noah Snavely's Bundler
// SfM system. This is also the camera model/residual for the bundle
// adjustment problems in the BAL dataset. It is templated so that we
// can use Ceres's automatic differentiation to compute analytic
// jacobians.
//
// For details see: http://phototour.cs.washington.edu/bundler/
// and http://grail.cs.washington.edu/projects/bal/
#ifndef CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
#define CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
#include "ceres/rotation.h"
#include <stdio.h>
using namespace std;
namespace ceres {
namespace examples {
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y, double f, double cx, double cy)
      : observed_x(observed_x), observed_y(observed_y), f(f), cx(cx), cy(cy) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = T(f) * p[0] / p[2] + T(cx);
    const T yp = T(f) * p[1] / p[2] + T(cy);

    const T predicted_x = xp;
    const T predicted_y = yp;


    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    // std::cout << "predicted: " << focal * distortion * xp-observed_x<< "  "<<focal * distortion * yp-observed_y << std::endl;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double f,
                                     const double cx,
                                     const double cy) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(new SnavelyReprojectionError(observed_x, observed_y, f, cx, cy)));
  }
  double observed_x;
  double observed_y;
  double f, cx, cy;
};

struct PoseSmoothnessError
{
    Eigen::Affine3d fromSrcToRefInv;
    Eigen::Affine3d refPoseInv;
    double rScale;
    double tScale;

    PoseSmoothnessError(const Eigen::Affine3d _fromSrcToRefInv, const Eigen::Affine3d _refPoseInv,
                        const double _rScale, const double _tScale):
    fromSrcToRefInv(_fromSrcToRefInv.matrix()), refPoseInv(_refPoseInv.matrix()), rScale(_rScale), tScale(_tScale)
    {}

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction* Create(const Eigen::Affine3d _fromSrcToRefInv, 
                                       const Eigen::Affine3d _refPoseInv,
                                       const double _rScale, 
                                       const double _tScale) {
        return (new ceres::AutoDiffCostFunction<PoseSmoothnessError, 6, 6>(new PoseSmoothnessError(_fromSrcToRefInv, _refPoseInv, _rScale, _tScale)));
    }

    template <typename T>
    bool operator()(const T* const camera, 
                          T* residuals) const
    {

        Eigen::Matrix<T, 3, 3> RSrcPose;
        ceres::AngleAxisToRotationMatrix(camera, RSrcPose.data());

        Eigen::Matrix<T, 4, 4> SrcPoseT;
        Eigen::Matrix<T, 4, 4> curFromSrcToRefT;
        Eigen::Matrix<T, 4, 4> posDifT;

        SrcPoseT.block(0, 0, 3, 3) = RSrcPose;
        SrcPoseT(0, 3) = camera[3];
        SrcPoseT(1, 3) = camera[4];
        SrcPoseT(2, 3) = camera[5];
        SrcPoseT(3, 0) = T(0);
        SrcPoseT(3, 1) = T(0);
        SrcPoseT(3, 2) = T(0);
        SrcPoseT(3, 3) = T(1.0);
        
        Eigen::Matrix<T, 4, 4> fromSrcToRefInvT;
        Eigen::Matrix<T, 4, 4> refPoseInvT;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                fromSrcToRefInvT(i,j) = T(fromSrcToRefInv(i,j));
                refPoseInvT(i,j) = T(refPoseInv(i,j));
            }
        }

        curFromSrcToRefT = refPoseInvT * SrcPoseT;
        posDifT = fromSrcToRefInvT * curFromSrcToRefT;

        Eigen::Matrix<T, 3, 3> posDifRT;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                posDifRT(i,j) = posDifT(i,j);
            }
        }

        T posDifInParams[6];
        ceres::RotationMatrixToAngleAxis(posDifRT.data(), posDifInParams);
        posDifInParams[3] = posDifT(0, 3);
        posDifInParams[4] = posDifT(1, 3);
        posDifInParams[5] = posDifT(2, 3);
        // cout << "posDifInParams:\n";

        residuals[0] = T(rScale)*posDifInParams[0];
        residuals[1] = T(rScale)*posDifInParams[1];
        residuals[2] = T(rScale)*posDifInParams[2];
        residuals[3] = T(tScale)*posDifInParams[3];
        residuals[4] = T(tScale)*posDifInParams[4];
        residuals[5] = T(tScale)*posDifInParams[5];

        return true;
    }
};

struct BinaryPoseSmoothnessError
{
    Eigen::Affine3d fromSrcToRef;
    double rScale;
    double tScale;

    BinaryPoseSmoothnessError(const Eigen::Affine3d _fromSrcToRef,
                        const double _rScale, const double _tScale):
    fromSrcToRef(_fromSrcToRef.matrix()), rScale(_rScale), tScale(_tScale)
    {}

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction* Create(const Eigen::Affine3d _fromSrcToRef,
                                        const double _rScale, const double _tScale)
    {
        return (new ceres::NumericDiffCostFunction<BinaryPoseSmoothnessError, ceres::CENTRAL, 6, 6, 6>(new BinaryPoseSmoothnessError(_fromSrcToRef, _rScale, _tScale)));
    }

    template <typename T>
    bool operator()(const T * const srcPoseParam, const T * const refPoseParam, T* residuals) const
    {

        Eigen::Matrix<T, 3, 3> srcPoseR;
        Eigen::Matrix<T, 3, 3> refPoseR;
        ceres::AngleAxisToRotationMatrix(srcPoseParam, srcPoseR.data());
        ceres::AngleAxisToRotationMatrix(refPoseParam, refPoseR.data());

        Eigen::Matrix<T, 4, 4> srcPoseT;
        Eigen::Matrix<T, 4, 4> refPoseT;
        Eigen::Matrix<T, 4, 4> curFromRefToSrcT;
        Eigen::Matrix<T, 4, 4> posDifT;

        srcPoseT.block(0, 0, 3, 3) = srcPoseR;
        srcPoseT(0, 3) = srcPoseParam[3];
        srcPoseT(1, 3) = srcPoseParam[4];
        srcPoseT(2, 3) = srcPoseParam[5];
        srcPoseT(3, 0) = T(0);
        srcPoseT(3, 1) = T(0);
        srcPoseT(3, 2) = T(0);
        srcPoseT(3, 3) = T(1.0);

        refPoseT.block(0, 0, 3, 3) = refPoseR;
        refPoseT(0, 3) = refPoseParam[3];
        refPoseT(1, 3) = refPoseParam[4];
        refPoseT(2, 3) = refPoseParam[5];
        refPoseT(3, 0) = T(0);
        refPoseT(3, 1) = T(0);
        refPoseT(3, 2) = T(0);
        refPoseT(3, 3) = T(1.0);
        Eigen::Matrix<T, 4, 4> fromSrcToRefT;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                fromSrcToRefT(i,j) = T(fromSrcToRef(i,j));
            }
        }

        curFromRefToSrcT = srcPoseT.inverse() * refPoseT;
        posDifT = fromSrcToRefT * curFromRefToSrcT;

        Eigen::Matrix<T, 3, 3> posDifRT;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                posDifRT(i,j) = posDifT(i,j);
            }
        }

        T posDifInParams[6];
        ceres::RotationMatrixToAngleAxis(posDifRT.data(), posDifInParams);
        posDifInParams[3] = posDifT(0, 3);
        posDifInParams[4] = posDifT(1, 3);
        posDifInParams[5] = posDifT(2, 3);
//        // cout << "posDifInParams:\n";

        residuals[0] = T(rScale)*posDifInParams[0];
        residuals[1] = T(rScale)*posDifInParams[1];
        residuals[2] = T(rScale)*posDifInParams[2];
        residuals[3] = T(tScale)*posDifInParams[3];
        residuals[4] = T(tScale)*posDifInParams[4];
        residuals[5] = T(tScale)*posDifInParams[5];

        return true;
    }
};


// Projection Error Only
struct SnavelyReprojectionOnlyError {
  SnavelyReprojectionOnlyError(double observed_x, double observed_y, 
                               double f, double cx, double cy, double* camera)
      : observed_x(observed_x), observed_y(observed_y), f(f), cx(cx), cy(cy) {
        for (int i = 0; i < 6; ++i)
        {
          cam[i] = camera[i];
        }
      }

  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T camera[6];
    for (int i = 0; i < 6; ++i)
    {
      camera[i] = T(cam[i]);
    }
    T p[3];
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    const T xp = T(f) * p[0] / p[2] + T(cx);
    const T yp = T(f) * p[1] / p[2] + T(cy);

    const T predicted_x = xp;
    const T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double f,
                                     const double cx,
                                     const double cy,
                                     double* camera) {
    // cout << residuals[0] << endl;
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionOnlyError, 2, 3>(new SnavelyReprojectionOnlyError(observed_x, observed_y, f, cx, cy, camera)));
  }
  double observed_x;
  double observed_y;
  double f, cx, cy;
  double cam[6];
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 10 parameters. 4 for rotation, 3 for
// translation, 1 for focal length and 2 for radial distortion. The
// principal point is not modeled (i.e. it is assumed be located at
// the image center).
struct SnavelyReprojectionErrorWithQuaternions {
  // (u, v): the position of the observation with respect to the image
  // center point.
  SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2,3] is are the rotation of the camera as a quaternion.
    //
    // We use QuaternionRotatePoint as it does not assume that the
    // quaternion is normalized, since one of the ways to run the
    // bundle adjuster is to let Ceres optimize all 4 quaternion
    // parameters without a local parameterization.
    T p[3];
    QuaternionRotatePoint(camera, point, p);
    p[0] += camera[4];
    p[1] += camera[5];
    p[2] += camera[6];
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = - p[0] / p[2];
    const T yp = - p[1] / p[2];
    // Apply second and fourth order radial distortion.
    const T& l1 = camera[8];
    const T& l2 = camera[9];
    const T r2 = xp*xp + yp*yp;
    const T distortion = 1.0 + r2  * (l1 + l2  * r2);
    // Compute final projected point position.
    const T& focal = camera[7];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<
            SnavelyReprojectionErrorWithQuaternions, 2, 10, 3>(
                new SnavelyReprojectionErrorWithQuaternions(observed_x,
                                                            observed_y)));
  }
  double observed_x;
  double observed_y;
};
}  // namespace examples
}  // namespace ceres
#endif  // CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_