# Ubuntu18.04下安装OpenCV4.0

下面先不要看,按照这个链接来安装:https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/

## 1. 安装依赖项

安装依赖项,下面是原文中提供的;

```bash
sudo apt-get -y install libopencv-dev build-essential cmake libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
```

由于出现出错,需要进行修改

```bash
将libpng12-dev改成libpng-dev  # 这是因为在ubuntu16.04后已经弃用
不过最后后面还是安装了libpng12-dev
```

```bash
E: Unable to locate package libjasper-dev

解决:
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt-get update
再执行一遍即可
```

```bash
E: Unable to locate package libgstreamer0.10-dev
好像是版本问题

sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

因此,最终需要执行的是:

```bash
sudo apt-get -y install libopencv-dev build-essential cmake libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev libqt4-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
```



## 2. 编译opencv4

```bash
wget "https://github.com/opencv/opencv/archive/4.0.0.tar.gz" -O opencv.tar.gz
wget "https://github.com/opencv/opencv_contrib/archive/4.0.0.tar.gz" -O opencv_contrib.tar.gz

tar -zxvf opencv.tar.gz
tar -zxvf opencv_contrib.tar.gz

cd opencv-4.0.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/chen/workspace/github/opencv4/opencv-4.0.0/build -D INSTALL_C_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=/home/chen/workspace/github/opencv4/opencv_contrib-4.0.0/modules ../

```

出错

```
No package ‘gtk+-3.0’ found

sudo apt-get install libgtk-3-dev
```

