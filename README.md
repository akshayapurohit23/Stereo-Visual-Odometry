## Synopsis
System is developed to achieve Visual Odometry for a moving car, using the output of a stereo camera. Rectified stereo images from a moving camera is used to estimate camera pose and reconstruct a 3D map of scene points. Our system is further optimized using bundle adjustment and pose-graph optimization. Developed model was used to estimate the trajectory of the camera for various scenarios in the KITTI dataset.

## Results

Visual Odometry results for two sequences from the KITTI dataset.

<img src="https://raw.githubusercontent.com/akshayapurohit23/Stereo-Visual-Odometry/master/assets/Images/Seq06.png" width=350 height=275 align="middle" >     <img src="https://raw.githubusercontent.com/akshayapurohit23/Stereo-Visual-Odometry/master/assets/Images/Seq09.png" width=350 height=275 align="middle" >

[![](https://img.youtube.com/vi/EFFH1OTh_IQ/0.jpg)](https://www.youtube.com/watch?v=EFFH1OTh_IQ)

## Dependencies:
```
  OpenCV2.4
  PCL
  g2o
```

## Building Project in terminal:
```
  mkdir build
  cd build
  cmake ..
  make
```

##  Running project
```
cd <project_dir>
./vo PATH_TO_LEFT_IMAGE_SET_DIRECTORY PATH_TO_RIGHT_IMAGE_SET_DIRECTORY PATH_TO_YAML_FILE
```

## Contributors
```
Xiaoyu Zhou @ucsdxiaoyuzhou
Akshaya Purohit @akshayapurohit23
Bolun Zhang @zblzcj
Leonard Melvix @lmelvix
```
