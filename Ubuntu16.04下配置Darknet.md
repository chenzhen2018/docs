# Ubuntu 16.04配置Darknet

`Ubuntu 18.04`, `Darknet(AlexeyAB)` 

```
# 依赖项
gcc
g++
opencv
cdua+cudnn

darknet
```

## 1. 安装依赖项

### 1.1 gcc

安装

```bash
sudo apt install gcc
```

查看版本信息

```bash
gcc --version
```

### 1.2 g++

安装

```bash
sudo apt install g++
```

查看版本

```bash
g++ --version
```

### 1.3 opencv

安装

```bash
sudo apt-get install libopencv-dev
```

卸载

```bash
sudo apt-get autoremove libopencv-dev
```

查看版本信息

```bash
pkg-config --modversion opencv
```

### 1.4 cuda+cudnn

省略

## 2. Darknet

```bash
git clone https://github.com/AlexeyAB/darknet.git
```

### 2.1 修改Makefile

```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=1
OPENMP=1
LIBSO=1
ZED_CAMERA=0
```

### 2.2 编译

```bash
cd darknet
make
```

### 2.3 下载yolov3.weights

```
https://pjreddie.com/media/files/yolov3.weights
```

### 2.4 测试

将`yolov3.weights`拷贝到`./darknet`下，执行

```
cd darknet 

./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights

./data/dog.jpg
```

