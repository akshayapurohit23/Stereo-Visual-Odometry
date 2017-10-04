## Dependnicies:
```
  OpenCV2.4
  PCL
  g2o
```
## Building Project in terminal:
```
  mkdir build cd build cmake .. make
```

##  Run go to project directory
```
./vo PATH_TO_LEFT_IMAGE_SET_DIRECTORY PATH_TO_RIGHT_IMAGE_SET_DIRECTORY PATH_TO_YAML_FILE
```

Note: In case that the terminal shows "./vo: error while loading shared libraries: libg2o_core.so: cannot open shared object file: No such file or directory" or similar error, do following:

in terminal: ```cp -i /etc/ld.so.conf ~/Desktop/ gedit ~/Desktop/ld.so.conf```

in gedit: add following line: "/usr/local/lib"

in terminal: ```sudo cp -i ~/Desktop/ld.so.conf /etc/ld.so.conf sudo ldconfig```

## Contributors
```
Akshaya Purohit
Xiaoyu Zhou
```
