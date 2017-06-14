FIND_PACKAGE( PCL REQUIRED)
FIND_PACKAGE( OpenCV 2.4 REQUIRED )

LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

#PCL
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

#g2o
#SET( G2O_ROOT /usr/local/include/g2o )
#FIND_PACKAGE( G2O REQUIRED )

#CSparse
#FIND_PACKAGE( CSparse REQUIRED )
#INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

# Eigen
FIND_PACKAGE( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR})

### CERES
FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
MESSAGE("ceres found ${Ceres_FOUND} at ${Ceres_DIR}")

### GFLAGS
FIND_PACKAGE(gflags REQUIRED)
INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIRS})
MESSAGE(${GFLAGS_INCLUDE_DIRS})
MESSAGE("gflags found ${Gflags_FOUND} at ${GFLAGS_INCLUDE_DIRS} with namespace: ${GFLAGS_NAMESPACE}")


#==build library for my own source code=====
add_library (${PROJECT_NAME} SHARED 
draw.cpp
Frame.cpp
Map.cpp
MapPoint.cpp
utils.cpp
system.cpp
Mapviewer.cpp
optimizer.cpp
)

target_link_libraries(${PROJECT_NAME}
${PCL_LIBRARIES} 
${OpenCV_LIBS}
${EIGEN3_LIBS}
#g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY}
${CERES_LIBRARIES} 
${GFLAGS_LIBRARIES}
)

#=======build main=========================
ADD_EXECUTABLE(ba main.cpp )
TARGET_LINK_LIBRARIES( 
ba 
${PROJECT_NAME} 
${OpenCV_LIBS}
${PCL_LIBRARIES}
#${CSPARSE_LIBRARY} 
${CERES_LIBRARIES} 
${GFLAGS_LIBRARIES}
)
