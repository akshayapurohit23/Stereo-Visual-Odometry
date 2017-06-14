#include "optimizer.hpp"

Optimizer::Optimizer(bool printOut, Problem& prob):globalBAProblem(prob){
    // BAProblem = problem;
    options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = printOut;
//    options.max_num_iterations = 10;
}

void Optimizer::poseOptimization(vector<Frame* >& frames) {

    Problem poseOptProblem;
    double* cameraParameter_ = new double[cameraBlkSize * frames.size()];
    //construct camera parameters
    for(int n = 0; n < frames.size(); n++) {
        cameraParameter_[cameraBlkSize*n + 0] = frames[n]->worldRvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 1] = frames[n]->worldRvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 2] = frames[n]->worldRvec.at<double>(2,0);

        cameraParameter_[cameraBlkSize*n + 3] = frames[n]->worldTvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 4] = frames[n]->worldTvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 5] = frames[n]->worldTvec.at<double>(2,0);
    }   
    double* cameras = cameraParameter_;
    double* camera = cameras;

    //add constrain
    for(int n = 0; n < frames.size(); n++) {
        for(auto obs : frames[n]->relativePose){
            camera = cameras + n*cameraBlkSize;
            double* refCamera = cameras + obs.first*cameraBlkSize;
            //add an edge for the current camera
            Eigen::Affine3d fromSrcToRef, refPose;
            fromSrcToRef = vectorToTransformation(obs.second.first, 
                                                  obs.second.second);

            Eigen::Affine3d fromSrcToRefInv;
            fromSrcToRefInv = fromSrcToRef.inverse();

            CostFunction* costFunc = BinaryPoseSmoothnessError::Create(fromSrcToRefInv, 
                                                                 		1.,1.);
            poseOptProblem.AddResidualBlock(costFunc, NULL, camera, refCamera);
        }   
    }

    Solve(options, &poseOptProblem, &summary);

    //update all poses
    for(int n = 0; n < frames.size(); n++){
        frames[n]->worldRvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 0];
        frames[n]->worldRvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 1];
        frames[n]->worldRvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 2];
        frames[n]->worldTvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 3];
        frames[n]->worldTvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 4];
        frames[n]->worldTvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 5];
    }

}

void Optimizer::globalBundleAdjustment(Map* slammap, vector<Frame*> frames){
    // set camera poses
    double* cameraParameter_ = new double[cameraBlkSize*frames.size()];
    cout << "frame size: " << frames.size() << endl;
    for(int n = 0; n < frames.size(); n++){
        cameraParameter_[cameraBlkSize*n + 0] = frames[n]->worldRvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 1] = frames[n]->worldRvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 2] = frames[n]->worldRvec.at<double>(2,0);

        cameraParameter_[cameraBlkSize*n + 3] = frames[n]->worldTvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 4] = frames[n]->worldTvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 5] = frames[n]->worldTvec.at<double>(2,0);
    }

    long unsigned int numPointsParameters = pointBlkSize*slammap->allMapPointNumber();
    double* pointParameters_;
    pointParameters_ = new double[numPointsParameters];
    //convert container of mappoints to vector
    vector<MapPoint*> vallMapPoints;
    //add all map points into parameter
    cout << "adding mappoints into problem...";
    int n = 0;
    for(set<MapPoint*>::iterator it = slammap->allMapPoints.begin(); 
                                 it!= slammap->allMapPoints.end(); 
                                 it++){
        vallMapPoints.push_back(*it);

        pointParameters_[pointBlkSize*n + 0] = (double)(*it)->pos.x;
        pointParameters_[pointBlkSize*n + 1] = (double)(*it)->pos.y;
        pointParameters_[pointBlkSize*n + 2] = (double)(*it)->pos.z;

        n++;
    }
    cout <<" done!" << endl;
    double* points = pointParameters_;
    double* cameras = cameraParameter_;
    double* point = points;
    double* camera = cameras;

    //add mappoints
    cout << "adding observations into problem...";
    long unsigned int count = 0;//used to count mappoints
    for(vector<MapPoint*>::iterator it = vallMapPoints.begin();
                                    it!= vallMapPoints.end();
                                    it++){
        //iterate all observations of a mappoint
        for(map<Frame*, unsigned int>::iterator pIt = (*it)->observations.begin();
                                                pIt!= (*it)->observations.end();
                                                pIt++){
            cv::Point2f observedPt;
            observedPt.x = pIt->first->keypointL[pIt->second].pt.x;
            observedPt.y = pIt->first->keypointL[pIt->second].pt.y;
            camera = cameras + cameraBlkSize*pIt->first->frameID;
            point = points + pointBlkSize*count;
            //if first camera, fix
            if(pIt->first->frameID == 0){
                CostFunction* costFunc = SnavelyReprojectionOnlyError::Create((double)observedPt.x, 
                                                                              (double)observedPt.y,
                                                                               pIt->first->fx,
                                                                               pIt->first->cx,
                                                                               pIt->first->cy,
                                                                               camera);
                globalBAProblem.AddResidualBlock(costFunc, NULL, point);
            }
            //else do not fix
            else{
                CostFunction* costFunc = SnavelyReprojectionError::Create((double)observedPt.x, 
                                                                          (double)observedPt.y,
                                                                          pIt->first->fx,
                                                                          pIt->first->cx,
                                                                          pIt->first->cy);
                globalBAProblem.AddResidualBlock(costFunc, NULL, camera, point);
            }
        }
        count++;
    }
    cout << "  done!" << endl;
    cout << "solving problem..." <<endl;
    Solve(options, &globalBAProblem, &summary);

    //update mappoints;
    n = 0;
    for(set<MapPoint*>::iterator it = slammap->allMapPoints.begin(); 
                                 it!= slammap->allMapPoints.end(); 
                                 it++){
        (*it)->pos.x = (float)pointParameters_[pointBlkSize*n + 0];
        (*it)->pos.y = (float)pointParameters_[pointBlkSize*n + 1];
        (*it)->pos.z = (float)pointParameters_[pointBlkSize*n + 2];

        n++;
    }
    for(int n = 0; n < frames.size(); n++){
        frames[n]->worldRvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 0];
        frames[n]->worldRvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 1];
        frames[n]->worldRvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 2];
        frames[n]->worldTvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 3];
        frames[n]->worldTvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 4];
        frames[n]->worldTvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 5];
    }

}


void Optimizer::reprojectionOnlyAdjustment(Map* slammap, vector<Frame*> frames){
  // set camera poses
    double* cameraParameter_ = new double[cameraBlkSize*frames.size()];
    cout << "frame size: " << frames.size() << endl;
    for(int n = 0; n < frames.size(); n++){
        cameraParameter_[cameraBlkSize*n + 0] = frames[n]->worldRvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 1] = frames[n]->worldRvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 2] = frames[n]->worldRvec.at<double>(2,0);

        cameraParameter_[cameraBlkSize*n + 3] = frames[n]->worldTvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 4] = frames[n]->worldTvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 5] = frames[n]->worldTvec.at<double>(2,0);
    }

    long unsigned int numPointsParameters = pointBlkSize*slammap->allMapPointNumber();
    double* pointParameters_;
    pointParameters_ = new double[numPointsParameters];
    //convert container of mappoints to vector
    vector<MapPoint*> vallMapPoints;
    //add all map points into parameter
    cout << "adding mappoints into problem...";
    int n = 0;
    for(set<MapPoint*>::iterator it = slammap->allMapPoints.begin(); 
                                 it!= slammap->allMapPoints.end(); 
                                 it++){
        vallMapPoints.push_back(*it);

        pointParameters_[pointBlkSize*n + 0] = (double)(*it)->pos.x;
        pointParameters_[pointBlkSize*n + 1] = (double)(*it)->pos.y;
        pointParameters_[pointBlkSize*n + 2] = (double)(*it)->pos.z;

        n++;
    }
    cout <<" done!" << endl;
    double* points = pointParameters_;
    double* cameras = cameraParameter_;
    double* point = points;
    double* camera = cameras;

    //add mappoints
    cout << "adding observations into problem...";
    long unsigned int count = 0;//used to count mappoints
    for(vector<MapPoint*>::iterator it = vallMapPoints.begin();
                                    it!= vallMapPoints.end();
                                    it++){
        //iterate all observations of a mappoint
        for(map<Frame*, unsigned int>::iterator pIt = (*it)->observations.begin();
                                                pIt!= (*it)->observations.end();
                                                pIt++){
            cv::Point2f observedPt;
            observedPt.x = pIt->first->keypointL[pIt->second].pt.x;
            observedPt.y = pIt->first->keypointL[pIt->second].pt.y;
            camera = cameras + cameraBlkSize*pIt->first->frameID;
            point = points + pointBlkSize*count;
            //fix all cameras
            CostFunction* costFunc = SnavelyReprojectionOnlyError::Create((double)observedPt.x, 
                                                                          (double)observedPt.y,
                                                                           pIt->first->fx,
                                                                           pIt->first->cx,
                                                                           pIt->first->cy,
                                                                           camera);
            globalBAProblem.AddResidualBlock(costFunc, NULL, point);
        
        }
        count++;
    }
    cout << "  done!" << endl;
    cout << "solving problem..." <<endl;
    Solve(options, &globalBAProblem, &summary);

    //update mappoints;
    n = 0;
    for(set<MapPoint*>::iterator it = slammap->allMapPoints.begin(); 
                                 it!= slammap->allMapPoints.end(); 
                                 it++){
        (*it)->pos.x = (float)pointParameters_[pointBlkSize*n + 0];
        (*it)->pos.y = (float)pointParameters_[pointBlkSize*n + 1];
        (*it)->pos.z = (float)pointParameters_[pointBlkSize*n + 2];
        n++;
    }

}



void Optimizer::solveProblem(Problem& pb){
    cout << "solving problem..." <<endl;
    Solve(options, &pb, &summary);
}


void Optimizer::localBundleAdjustment(vector<Frame* > frames, int startIdx, int length){
    unsigned int minFrameID = frames[startIdx]->frameID;
    unsigned int maxFrameID = frames[MIN(startIdx+length-1, frames.size()-1)]->frameID;

    //get a deep copy of frames
    vector<Frame> localFrames;
    for(int n = minFrameID; n < maxFrameID+1; n++){
        localFrames.push_back(*frames[n]);
    }

    //construct camera parameters
    double* cameraParameter_ = new double[6*(maxFrameID-minFrameID+1)];

    for(int n = minFrameID; n < maxFrameID+1; n++){//only store poses need to be adjusted
        cameraParameter_[cameraBlkSize*(n-minFrameID) + 0] = localFrames[n-minFrameID].worldRvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*(n-minFrameID) + 1] = localFrames[n-minFrameID].worldRvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*(n-minFrameID) + 2] = localFrames[n-minFrameID].worldRvec.at<double>(2,0);

        cameraParameter_[cameraBlkSize*(n-minFrameID) + 3] = localFrames[n-minFrameID].worldTvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*(n-minFrameID) + 4] = localFrames[n-minFrameID].worldTvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*(n-minFrameID) + 5] = localFrames[n-minFrameID].worldTvec.at<double>(2,0);
    }
    // build problem
    Problem localProblem;

    vector<MapPoint> localMapPoints; //not using pointer
    unordered_set<MapPoint* > localMapPointsSet;
    vector<cv::Point3f> localScenePoints;
    vector<int> localScenePointsIdx2;
    vector<int> localScenePointsIdx3;

    vector<cv::Point2f> localKeypoints;
    vector<cv::Point2f> localKeypoints2;
    vector<cv::Point2f> localKeypoints3;

    double* cameras = cameraParameter_;
    double* camera = cameras;

    int offset = 0;

    //============ original method ===============================================
    //first iteration to create vector of point parameters
    //set all mappoints in the first frame as parameter potins, temporarily store into a vector
    // for(int p = 0; p < localFrames[0].keypointL.size(); p++){
    //     if(localFrames[0].mappoints[p] != NULL && !localFrames[0].mappoints[p]->isBad){
    //         localMapPoints.push_back(*localFrames[0].mappoints[p]);
    //         localScenePoints.push_back(localFrames[0].mappoints[p]->pos);
    //     }
    // }
    //=========== original method ends ============================================


    //============= test 2: add all points into parameters ========================
    for(int f = 0; f < localFrames.size(); f++){
        for(int p = 0; p < localFrames[f].mappoints.size(); p++){
            if(localFrames[f].mappoints[p] != NULL && !localFrames[f].mappoints[p]->isBad){
                localMapPointsSet.insert(localFrames[f].mappoints[p]);
            }
        }
    }
    //============= test 2 ends ===================================================

    //only for display
    // for(int n = 0; n < frames[MAX(0,MIN(startIdx+offset,frames.size()-1))]->keypointL.size(); n++){
    //     if(frames[MAX(0,MIN(startIdx+offset,frames.size()-1))]->mappoints[n] != NULL &&
    //        !frames[MAX(0,MIN(startIdx+offset,frames.size()-1))]->mappoints[n]->isBad){
    //         localKeypoints.push_back(frames[MAX(0,MIN(startIdx+offset,frames.size()-1))]->keypointL[n].pt);
    //     }
    // }

    //create point parameters
    long unsigned int numPointsParameters = pointBlkSize*localMapPointsSet.size();
    double* pointParameters_;
    pointParameters_ = new double[numPointsParameters];

    //add all mappoints into problem
    int ptCount = 0;

    //============ original method ===============================================
    // //set all mappoints in the first frame as parameter potins, temporarily store into a vector
    // for(int p = 0; p < localFrames[0].keypointL.size(); p++){
    //     if(localFrames[0].mappoints[p] != NULL && !localFrames[0].mappoints[p]->isBad){

    //         pointParameters_[pointBlkSize*ptCount + 0] = (double)localFrames[0].mappoints[p]->pos.x;
    //         pointParameters_[pointBlkSize*ptCount + 1] = (double)localFrames[0].mappoints[p]->pos.y;
    //         pointParameters_[pointBlkSize*ptCount + 2] = (double)localFrames[0].mappoints[p]->pos.z;
    //         ptCount++;
    //     }
    // }
    //=========== original method ends =============================================

    //============= test 2: add all points into parameters =========================
    for(unordered_set<MapPoint* >::iterator pIt = localMapPointsSet.begin();
                                            pIt != localMapPointsSet.end();
                                            pIt++){
        pointParameters_[pointBlkSize*ptCount + 0] = (double)(*pIt)->pos.x;
        pointParameters_[pointBlkSize*ptCount + 1] = (double)(*pIt)->pos.y;
        pointParameters_[pointBlkSize*ptCount + 2] = (double)(*pIt)->pos.z;
        ptCount++;
    }
    //============= test 2 ends ===================================================

    //add observations
    double* points = pointParameters_;
    double* point = points;
    //====test==============================================================================
    // Mat Rvec = Mat::zeros(3,1,CV_64F);
    // Mat Tvec = Mat::zeros(3,1,CV_64F);

    // Rvec.at<double>(0,0) = cameraParameter_[0];
    // Rvec.at<double>(1,0) = cameraParameter_[1];
    // Rvec.at<double>(2,0) = cameraParameter_[2];

    // Tvec.at<double>(0,0) = cameraParameter_[3];
    // Tvec.at<double>(1,0) = cameraParameter_[4];
    // Tvec.at<double>(2,0) = cameraParameter_[5];

    // vector<Point2f> projImgPts;
    // Mat K = frames[0]->srp.P1.colRange(0,3).clone();
    // projectPoints(localScenePoints, Rvec, Tvec, K, Mat(), projImgPts);
    // drawFarandCloseFeatures(localFrames[0].imgL,
    //                         localKeypoints,
    //                         projImgPts,"before");
    //====test end===========================================================================
    vector<cv::Point2f> observedPts;
    int pointCount = 0;
    for(unordered_set<MapPoint*>::iterator mappointIt = localMapPointsSet.begin();
                                           mappointIt != localMapPointsSet.end();
                                           mappointIt++){
        for(map<Frame*, unsigned int>::iterator pIt = (*mappointIt)->observations.begin();
                                                pIt!= (*mappointIt)->observations.end();
                                                pIt++){
            cv::Point2f observedPt;

            observedPt.x = pIt->first->keypointL[pIt->second].pt.x;
            observedPt.y = pIt->first->keypointL[pIt->second].pt.y;

            point = points + pointBlkSize*pointCount;

            //only fix the first frame in the entire sequence
            if(pIt->first->frameID == 0 && startIdx< 1){
            // if(pIt->first->frameID == 0){
                camera = cameras;
                CostFunction* costFunc = SnavelyReprojectionOnlyError::Create((double)observedPt.x,
                                                                              (double)observedPt.y,
                                                                               pIt->first->fx,
                                                                               pIt->first->cx,
                                                                               pIt->first->cy,
                                                                               camera);
                localProblem.AddResidualBlock(costFunc, NULL, point);
            }
            //else do not fix
            else if(pIt->first->frameID >= minFrameID &&
                    pIt->first->frameID <= maxFrameID){
                camera = cameras + cameraBlkSize*(pIt->first->frameID - minFrameID);
                // ==== this part is to extract keypoint in the corresponding frame
                // ==== only for evaluation
                if(pIt->first->frameID == minFrameID + 1){
                    localScenePointsIdx2.push_back(pointCount);
                    localKeypoints2.push_back(observedPt);
                }
                if(pIt->first->frameID == minFrameID + 2){
                    localScenePointsIdx3.push_back(pointCount);
                    localKeypoints3.push_back(observedPt);
                }
                //========= test end ======================================

                CostFunction* costFunc = SnavelyReprojectionError::Create((double)observedPt.x,
                                                                          (double)observedPt.y,
                                                                          pIt->first->fx,
                                                                          pIt->first->cx,
                                                                          pIt->first->cy);
                localProblem.AddResidualBlock(costFunc, NULL, camera, point);
            }
        }
        pointCount++;
    }

    Solve(options, &localProblem, &summary);

    if(options.minimizer_progress_to_stdout){
        cout << summary.FullReport() << endl;
    }

    //update points and pose
    for(int n = minFrameID; n < maxFrameID+1; n++){
        frames[n]->worldRvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 0];
        frames[n]->worldRvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 1];
        frames[n]->worldRvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 2];

        frames[n]->worldTvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 3];
        frames[n]->worldTvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 4];
        frames[n]->worldTvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 5];
    }

    pointCount = 0;
    for(unordered_set<MapPoint* >::iterator pIt = localMapPointsSet.begin();
                                            pIt != localMapPointsSet.end();
                                            pIt++){
        (*pIt)->pos.x = (float)pointParameters_[pointBlkSize*pointCount + 0];
        (*pIt)->pos.y = (float)pointParameters_[pointBlkSize*pointCount + 1];
        (*pIt)->pos.z = (float)pointParameters_[pointBlkSize*pointCount + 2];
        pointCount++;
    }


    /*
    vector<Point2f> observedPts;

    for(int n = 0; n < localMapPoints.size(); n++){
        for(map<Frame*, unsigned int>::iterator pIt = localMapPoints[n].observations.begin();
                                                pIt!= localMapPoints[n].observations.end();
                                                pIt++){
            Point2f observedPt;

            observedPt.x = pIt->first->keypointL[pIt->second].pt.x;
            observedPt.y = pIt->first->keypointL[pIt->second].pt.y;

            point = points + pointBlkSize*n;

            //only fix the first frame in the entire sequence
            if(pIt->first->frameID == 0){
                camera = cameras;
                CostFunction* costFunc = SnavelyReprojectionOnlyError::Create((double)observedPt.x,
                                                                              (double)observedPt.y,
                                                                               pIt->first->fx,
                                                                               pIt->first->cx,
                                                                               pIt->first->cy,
                                                                               camera);
                localProblem.AddResidualBlock(costFunc, NULL, point);
            }
            //else do not fix
            else if(pIt->first->frameID >= minFrameID &&
                    pIt->first->frameID <= maxFrameID){
                camera = cameras + cameraBlkSize*(pIt->first->frameID - minFrameID);
                // ==== this part is to extract keypoint in the corresponding frame
                // ==== only for evaluation
                if(pIt->first->frameID == minFrameID + 1){
                    localScenePointsIdx2.push_back(n);
                    localKeypoints2.push_back(observedPt);
                }
                if(pIt->first->frameID == minFrameID + 2){
                    localScenePointsIdx3.push_back(n);
                    localKeypoints3.push_back(observedPt);
                }
                //========= test end ======================================

                CostFunction* costFunc = SnavelyReprojectionError::Create((double)observedPt.x,
                                                                          (double)observedPt.y,
                                                                          pIt->first->fx,
                                                                          pIt->first->cx,
                                                                          pIt->first->cy);
                localProblem.AddResidualBlock(costFunc, NULL, camera, point);
            }
        }
    }

    Solve(options, &localProblem, &summary);

    if(options.minimizer_progress_to_stdout){
        cout << summary.FullReport() << endl;
    }

    //update points and pose
    for(int n = minFrameID; n < maxFrameID+1; n++){
        frames[n]->worldRvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 0];
        frames[n]->worldRvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 1];
        frames[n]->worldRvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 2];

        frames[n]->worldTvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 3];
        frames[n]->worldTvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 4];
        frames[n]->worldTvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*(n-minFrameID) + 5];
    }

    unsigned int pointCount = 0;
    for(int p = 0; p < frames[startIdx]->keypointL.size(); p++){
         if(frames[startIdx]->mappoints[p] != NULL && !frames[startIdx]->mappoints[p]->isBad){

            frames[startIdx]->mappoints[p]->pos.x = (float)pointParameters_[pointBlkSize*pointCount + 0];
            frames[startIdx]->mappoints[p]->pos.y = (float)pointParameters_[pointBlkSize*pointCount + 1];
            frames[startIdx]->mappoints[p]->pos.z = (float)pointParameters_[pointBlkSize*pointCount + 2];

            pointCount++;
        }
    }
*/

    /*
//================================== display for evaluation ========================================
    Rvec.at<double>(0,0) = cameras[cameraBlkSize*(offset) + 0];
    Rvec.at<double>(1,0) = cameras[cameraBlkSize*(offset) + 1];
    Rvec.at<double>(2,0) = cameras[cameraBlkSize*(offset) + 2];

    Tvec.at<double>(0,0) = cameras[cameraBlkSize*(offset) + 3];
    Tvec.at<double>(1,0) = cameras[cameraBlkSize*(offset) + 4];
    Tvec.at<double>(2,0) = cameras[cameraBlkSize*(offset) + 5];

    vector<cv::Point3f> adjustedPts;
    for(int n = 0; n < localMapPoints.size(); n++){
        cv::Point3f pt;

        pt.x = (float)pointParameters_[pointBlkSize*n + 0];
        pt.y = (float)pointParameters_[pointBlkSize*n + 1];
        pt.z = (float)pointParameters_[pointBlkSize*n + 2];

        adjustedPts.push_back(pt);
    }
    vector<cv::Point2f> newProjPts;
    projectPoints(adjustedPts, Rvec, Tvec, K, cv::Mat(), newProjPts);

    float totalErr = 0;
    for(int n = 0; n < newProjPts.size(); n++){
        float dx = newProjPts[n].x - localKeypoints[n].x;
        float dy = newProjPts[n].y - localKeypoints[n].y;
        float tempErr = sqrt(dx*dx + dy*dy);
        totalErr += tempErr;

    }
//    cout << "total error: " << totalErr << endl;
    drawFarandCloseFeatures(localFrames[0].imgL,
                            localKeypoints,
                            newProjPts,"after");*/
//
//    //======================== 1 =============================
//    offset = 1;
//
//    Rvec.at<double>(0,0) = cameras[cameraBlkSize*(offset) + 0];
//    Rvec.at<double>(1,0) = cameras[cameraBlkSize*(offset) + 1];
//    Rvec.at<double>(2,0) = cameras[cameraBlkSize*(offset) + 2];
//
//    Tvec.at<double>(0,0) = cameras[cameraBlkSize*(offset) + 3];
//    Tvec.at<double>(1,0) = cameras[cameraBlkSize*(offset) + 4];
//    Tvec.at<double>(2,0) = cameras[cameraBlkSize*(offset) + 5];
//
//    adjustedPts.clear();
//    for(int n = 0; n < localScenePointsIdx2.size(); n++){
//        Point3f pt;
//        pt.x = (float)pointParameters_[pointBlkSize*localScenePointsIdx2[n] + 0];
//        pt.y = (float)pointParameters_[pointBlkSize*localScenePointsIdx2[n] + 1];
//        pt.z = (float)pointParameters_[pointBlkSize*localScenePointsIdx2[n] + 2];
//
//        adjustedPts.push_back(pt);
//    }
//    newProjPts.clear();
//    projectPoints(adjustedPts, Rvec, Tvec, K, Mat(), newProjPts);
//
//    totalErr = 0;
//    for(int n = 0; n < newProjPts.size(); n++){
//        float dx = newProjPts[n].x - localKeypoints2[n].x;
//        float dy = newProjPts[n].y - localKeypoints2[n].y;
//        float tempErr = sqrt(dx*dx + dy*dy);
//        totalErr += tempErr;
//
//    }
////    cout << "total error: " << totalErr << endl;
//    if(localKeypoints2.size() > 0){
//        drawFarandCloseFeatures(frames[MIN(startIdx+offset,frames.size()-1)]->imgL,
//                                localKeypoints2,
//                                newProjPts,"1");
//    }
//
//    //======================== 2 =============================
//    offset=2;
//
//    Rvec.at<double>(0,0) = cameras[cameraBlkSize*(offset) + 0];
//    Rvec.at<double>(1,0) = cameras[cameraBlkSize*(offset) + 1];
//    Rvec.at<double>(2,0) = cameras[cameraBlkSize*(offset) + 2];
//
//    Tvec.at<double>(0,0) = cameras[cameraBlkSize*(offset) + 3];
//    Tvec.at<double>(1,0) = cameras[cameraBlkSize*(offset) + 4];
//    Tvec.at<double>(2,0) = cameras[cameraBlkSize*(offset) + 5];
//
//    adjustedPts.clear();
//
//    for(int n = 0; n < localScenePointsIdx3.size(); n++){
//        Point3f pt;
//
//        pt.x = (float)pointParameters_[pointBlkSize*localScenePointsIdx3[n] + 0];
//        pt.y = (float)pointParameters_[pointBlkSize*localScenePointsIdx3[n] + 1];
//        pt.z = (float)pointParameters_[pointBlkSize*localScenePointsIdx3[n] + 2];
//
//        adjustedPts.push_back(pt);
//    }
//    newProjPts.clear();
//    if(adjustedPts.size() > 0){
//        projectPoints(adjustedPts, Rvec, Tvec, K, Mat(), newProjPts);
//
//        totalErr = 0;
//        for(int n = 0; n < newProjPts.size(); n++){
//            float dx = newProjPts[n].x - localKeypoints3[n].x;
//            float dy = newProjPts[n].y - localKeypoints3[n].y;
//            float tempErr = sqrt(dx*dx + dy*dy);
//            totalErr += tempErr;
//
//        }
//
////        cout << "total error: " << totalErr << endl;
//        drawFarandCloseFeatures(frames[MIN(startIdx+offset,frames.size()-1)]->imgL,
//                                localKeypoints3,
//                                newProjPts,"2");
//    }

delete [] cameraParameter_;
}

