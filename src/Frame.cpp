#include "Frame.h"
#include "utils.h"

using namespace cv;
using namespace std;


Frame::Frame(string leftImgFile, string rightImgFile, 
             STEREO_RECTIFY_PARAMS _srp, int _id, Map* _map):
			 frameID(_id), map(_map), srp(_srp){
    imgL = imread(leftImgFile);
    imgR = imread(rightImgFile);
    if(!imgL.data || ! imgR.data){
        cout << "image does not exist..." << endl;
        exit;
    }

    fx = _srp.P1.at<double>(0,0);
    fy = _srp.P1.at<double>(1,1);
    cx = _srp.P1.at<double>(0,2);
    cy = _srp.P1.at<double>(1,2);
    b = -_srp.P2.at<double>(0,3)/fx;

    SurfFeatureDetector detector(10, 6, 3);
    SurfDescriptorExtractor descriptor;
    
    detector.detect(imgL, keypointL);
    detector.detect(imgR, keypointR);
    
    Mat tempDespL;
    descriptor.compute(imgL, keypointL, tempDespL);
    descriptor.compute(imgR, keypointR, despR);

    rvec = Mat::zeros(3, 1, CV_64F);
    tvec = Mat::zeros(3, 1, CV_64F);

    worldRvec = Mat::zeros(3, 1, CV_64F);
    worldTvec = Mat::zeros(3, 1, CV_64F);
    vector<KeyPoint> stereoKeypointLeft, stereoKeypointRight;
    vector<DMatch> stereoMatches;

    //compute stereo matches
    vector<int> sceneInliers;
    matchFeatureKNN(tempDespL, despR, keypointL, keypointR,
    				stereoKeypointLeft, stereoKeypointRight,
    				stereoMatches, 0.8);

    compute3Dpoints(stereoKeypointLeft, stereoKeypointRight,
    				keypointL, keypointR, sceneInliers);

    vector<int> keypointLinliers;
    for(int idx : sceneInliers){
        keypointLinliers.push_back(stereoMatches[idx].queryIdx);
    }

    // extract descriptors
    despL = Mat(keypointLinliers.size(), tempDespL.cols, tempDespL.type());

    for(int n = 0; n < keypointLinliers.size(); n++){
        tempDespL.row(keypointLinliers[n]).copyTo(despL.row(n));
    }
    mappoints = vector<MapPoint*>(keypointL.size(), static_cast<MapPoint*>(NULL));
    originality = vector<bool>(keypointL.size(), false);
}

bool idx_comparator(const DMatch& m1, const DMatch& m2){
	return m1.queryIdx < m2.queryIdx;
}

void Frame::matchFrame(Frame* frame, bool useMappoints){
    rvec.release();
    tvec.release();

    matchesBetweenFrame.clear();
	//first step, match using features
    vector<KeyPoint> matchedPrev, matchedCurr;
    vector<DMatch> matches;
    matchFeatureKNN(despL, frame->despL, 
                    keypointL, frame->keypointL,
                    matchedPrev, matchedCurr,
                    matches, 0.8);

    // obtain obj_pts, img_pts and matchedIdx
    vector<Point3f> obj_pts;
    vector<Point2f> img_pts;
    vector<DMatch> tempMatches;
    for(auto m : matches){
        if(mappoints[m.queryIdx]!=NULL && 
           !mappoints[m.queryIdx]->isBad &&
           useMappoints){
            MapPoint* curMp = mappoints[m.queryIdx];
            obj_pts.push_back(curMp->getPositionInCameraCoordinate(worldRvec, worldTvec));
        }
        else{
            obj_pts.push_back(scenePts[m.queryIdx]);
        }
        
        img_pts.push_back(frame->keypointL[m.trainIdx].pt);
    }
    Mat inliers;
    PnP(obj_pts, img_pts, inliers);
    for(int n = 0; n < inliers.rows; n++){
        matchesBetweenFrame.push_back(matches[inliers.at<int>(n,0)]);
    }

    frame->rvec = rvec.clone();
    frame->tvec = tvec.clone();
    sort(matchesBetweenFrame.begin(), matchesBetweenFrame.end(), idx_comparator);
    vector<Point2f> finalP1, finalP2;
    for(int n = 0; n < matchesBetweenFrame.size(); n++){
        finalP1.push_back(keypointL[matchesBetweenFrame[n].queryIdx].pt);
        finalP2.push_back(frame->keypointL[matchesBetweenFrame[n].trainIdx].pt);
    }

    drawMatch(imgL, finalP1, finalP2, 1, "pnp inliers");
}

void Frame::setWrdTransVectorAndTransScenePts(cv::Mat _worldRvec, cv::Mat _worldTvec) {
    worldRvec = _worldRvec.clone();
    worldTvec = _worldTvec.clone();
    transformScenePtsToWorldCoordinate();
}

void Frame::transformScenePtsToWorldCoordinate(){
	Eigen::Affine3d worldTrans = vectorToTransformation(worldRvec, worldTvec);
	scenePtsinWorld = transformPoints(worldTrans, scenePts);
}

void Frame::manageMapPoints(Frame* frame){    
    int newMappointCount = 0;
    int newObservationCount = 0;
    for(int n = 0; n < matchesBetweenFrame.size(); n++){
    	unsigned int qIdx = (unsigned int)matchesBetweenFrame[n].queryIdx;
    	unsigned int tIdx = (unsigned int)matchesBetweenFrame[n].trainIdx;
    	if(mappoints[qIdx] == NULL && frame->mappoints[tIdx] == NULL){
    		//create new mappoint in current frame
    		//create a pointer to this mappoint in the matched frame
    		//also need to add an observation
    		mappoints[qIdx] = createNewMapPoint(qIdx);
            pointToExistingMapPoint(frame, mappoints[qIdx], tIdx);

    		mappoints[qIdx]->addObservation(frame, tIdx);
    		mappoints[qIdx]->addObservation(this, qIdx);

    		map->addMapPoint(mappoints[qIdx]);
    	}
        else if(mappoints[qIdx] == NULL && frame->mappoints[tIdx] != NULL){
            pointToExistingMapPoint(this, frame->mappoints[tIdx], qIdx);
            frame->mappoints[tIdx]->addObservation(this, qIdx);
        }
        else if(mappoints[qIdx] != NULL && frame->mappoints[tIdx] == NULL){
            pointToExistingMapPoint(frame, mappoints[qIdx], tIdx);
            mappoints[qIdx]->addObservation(frame,tIdx);
        }

    	else if(mappoints[qIdx] != NULL && frame->mappoints[tIdx] != NULL){
    		//add observation to existing mappoint
//    		mappoints[qIdx]->addObservation(frame, tIdx);
//            pointToExistingMapPoint(frame, mappoints[qIdx], tIdx);
//    		newObservationCount++;
    	}
    }
}

void Frame::addEdgeConstrain(unsigned int id, Mat relativeRvec, Mat relativeTvec){
    relativePose.insert(make_pair(id, make_pair(relativeRvec, relativeTvec)));
}


void Frame::judgeBadPoints(){
    //count total number
    int validPointNumber = 0;
    for(auto mappoint : mappoints){
        if(mappoint != NULL){validPointNumber++;}
    }
    //compute each point to other points average distance, if too large, discard
    for(auto mappoint : mappoints){
        if(mappoint != NULL) {
            int closeCount = 0, farCount = 0;
            for (auto compareMappoint : mappoints) {
                if (compareMappoint != NULL) {
                    float distX = compareMappoint->pos.x - mappoint->pos.x;
                    float distY = compareMappoint->pos.y - mappoint->pos.y;
                    float distZ = compareMappoint->pos.z - mappoint->pos.z;
                    if(sqrt(pow(distX, 2) + pow(distY, 2) + pow(distZ, 2)) < 100 * b){
                        closeCount++;
                    }
                    else{
                        farCount++;
                    }
                }
            }
            if ((float)farCount/(float)closeCount > 0.1) { mappoint->isBad = true; }
        }
    }
}

void Frame::judgeBadPointsKdTree() {
    //build kd-tree
    int validPointNumber = 0;
    vector<Point3f> mappoints3f;
    for(auto mappoint : mappoints) {
        if(mappoint != NULL) {
            mappoints3f.push_back(mappoint->pos);
            validPointNumber++;
        }
    }

    KDTree pointsKDTree(mappoints3f, false);
    int K =10, Emax = INT_MAX;
    int idx[K];
    float dist[K];
    
    // for(auto mappoint3f : mappoints3f) {
    //     pointsKDTree.findNearest(mappoint3f, K, Emax, idx, 0, dist);
    //     float aveDist = 0;
    //     for(int n = 0; n < K; n++) {aveDist += dist[n];}
    //     aveDist /= (float)K;
    //     cout << "averag distance: " << aveDist << endl;

    // }
}



MapPoint* Frame::createNewMapPoint(unsigned int pointIdx){
	return new MapPoint(scenePtsinWorld[pointIdx], frameID, pointIdx);
}

void Frame::pointToExistingMapPoint(Frame* frame, MapPoint* mp, unsigned int currIdx){
	frame->mappoints[currIdx] = mp;
}

void Frame::PnP(vector<Point3f> obj_pts, 
                vector<Point2f> img_pts,
                Mat& inliers){

     //solve PnP
    Mat K = srp.P2.colRange(0,3).clone();
    // Mat temprvec, temptvec;
    if(obj_pts.size() == 0){
        cout << "points for PnP is 0, cannot solve." << endl;
        return;
    }
    solvePnPRansac(obj_pts, img_pts, K, Mat(),
                   rvec, tvec, false, 2000,3.0, 300, inliers);
}

void Frame::releaseMemory(){
    imgL.release();
    imgR.release();
    despL.release();
    despR.release();

}

void Frame::matchFeatureKNN(const Mat& desp1, const Mat& desp2, 
                            const vector<KeyPoint>& keypoint1, 
                            const vector<KeyPoint>& keypoint2,
                            vector<KeyPoint>& matchedKeypoint1,
                            vector<KeyPoint>& matchedKeypoint2,
                            vector<DMatch>& matches,
                            double knn_match_ratio,
                            bool hamming){

    matchedKeypoint1.clear();
    matchedKeypoint2.clear();
    matches.clear();

    float imgThres = 0.15 * std::sqrt(std::pow(imgL.rows, 2)+std::pow(imgL.cols, 2));

    if(hamming){
    	cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming(2)");
    	vector< vector<cv::DMatch> > matches_knn;
	    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
	    vector< cv::DMatch > tMatches;

	    for ( size_t i=0; i<matches_knn.size(); i++ )
	    {
	        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
	            tMatches.push_back( matches_knn[i][0] );
	    }
	    
	    if (tMatches.size() <= 20) //too few matches
	        return;

	    vector<KeyPoint> tMatchedKeypoint1, tMatchedKeypoint2;
	    for ( auto m:tMatches )
	    {
	        Point2f pt1, pt2;
	        pt1 = keypoint1[m.queryIdx].pt;
	        pt2 = keypoint2[m.trainIdx].pt;
	        float ptdist = std::sqrt(std::pow(pt1.x-pt2.x, 2) + std::pow(pt1.y-pt2.y, 2));
	        if(ptdist < imgThres){
	            matchedKeypoint1.push_back(keypoint1[m.queryIdx]);
	            matchedKeypoint2.push_back(keypoint2[m.trainIdx]);
	            matches.push_back(m);
	        } 
	    }  
    }
    else{
    	cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce");
	    vector< vector<cv::DMatch> > matches_knn;
	    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
	    vector< cv::DMatch > tMatches;

	    for ( size_t i=0; i<matches_knn.size(); i++ )
	    {
	        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
	            tMatches.push_back( matches_knn[i][0] );
	    }
	    
	    if (tMatches.size() <= 20) //too few matches
	        return;

	    vector<KeyPoint> tMatchedKeypoint1, tMatchedKeypoint2;
	    for ( auto m:tMatches )
	    {
	        Point2f pt1, pt2;
	        pt1 = keypoint1[m.queryIdx].pt;
	        pt2 = keypoint2[m.trainIdx].pt;
	        float ptdist = std::sqrt(std::pow(pt1.x-pt2.x, 2) + std::pow(pt1.y-pt2.y, 2));
	        if(ptdist < imgThres){
	            matchedKeypoint1.push_back(keypoint1[m.queryIdx]);
	            matchedKeypoint2.push_back(keypoint2[m.trainIdx]);
	            matches.push_back(m);
	        } 
	    } 
    } 
}

void Frame::compute3Dpoints(vector<KeyPoint>& kl, 
					 	    vector<KeyPoint>& kr,
					 		vector<KeyPoint>& trikl,
					 		vector<KeyPoint>& trikr,
                            vector<int>& inliers){

	vector<KeyPoint> copy_kl, copy_kr;
	copy_kl = kl;
	copy_kr = kr;

	scenePts.clear();
	trikl.clear();
	trikr.clear();
    inliers.clear();
	double thres = 45*b;
	for(int n = 0; n < copy_kl.size(); n++){
		Point3f pd;
		double d = fabs(copy_kl[n].pt.x - copy_kr[n].pt.x);
		pd.x = (float)(b*(copy_kl[n].pt.x - cx)/d);
		pd.y = (float)(b*(copy_kl[n].pt.y - cy)/d);
		pd.z = (float)(b*fx/d);
		if(pd.z < thres &&
		   fabs(copy_kl[n].pt.y - copy_kr[n].pt.y) < 4){
			scenePts.push_back(pd);
			trikl.push_back(copy_kl[n]);
			trikr.push_back(copy_kr[n]);
            inliers.push_back(n);
		}
	}
}

Eigen::Affine3d Frame::getWorldTransformationMatrix(){
    Mat R;
    Eigen::Affine3d result;
    Rodrigues(worldRvec, R);
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            result(r,c) = R.at<double>(r,c);
        }
    }

    result(0,3) = worldTvec.at<double>(0,0);
    result(1,3) = worldTvec.at<double>(1,0);
    result(2,3) = worldTvec.at<double>(2,0);
    return result;
}
