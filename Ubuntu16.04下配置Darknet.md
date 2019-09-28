## Ubuntu 16.04配置Darknet





### 1. 下载darknet

下载地址：https://github.com/AlexeyAB/darknet

### 2. 安装CMake

```
# 参考
https://www.cnblogs.com/TOLLA/p/9647279.html
```

```
https://cmake.org/files/
```

在上述路径下选择合适的CMake版本

我这里选择/v3.14/cmake-3.14.5-Linux-86_64.tar.gz

下载到Downloads下，解压后；

```
cd Downloads
sudo mv cmake-3.14.5-Linux-86_64 /opt/
```

配置~/.bashrc

```
sudo gedit ~/.bashrc
```

在最后添加

```
export PATH=/opt/cmake-3.14.5-Linux-x86_64/bin:$PATH
```

保存，退出；执行

```
source ~/.bashrc
```

测试：

```
cmake --version
```

输出

```
cmake version 3.14.5

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

### 3. CUDA+cuDNN

已配置

### 4. 配置OpenCV

查看opencv版本

```bash
pkg-config --modversion opencv
```

#### 4.1 apt-get

```
# install
sudo apt-get install libopencv-dev

# uninstall
sudo apt-get autoremove libopencv-dev
```

```
/home/louis/Downloads/opencv-2.4.9
mkdir build
cd build
cmake  /home/louis/Downloads/opencv-2.4.9
make                                          //无尽的等待
sudo make install
```



#### 4.2 源码编译(未完成)

```
# 官方文档
https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html
```

##### 4.2.1 安装依赖项

```
# 检查
GCC 4.4.x or later
CMake 2.8.7 or higher
Git
```

```bash
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

##### 4.2.2 下载

下载源码

```
https://opencv.org/releases/
```

##### 4.2.3 编译

```bash
cd ~/opencv-4.1.0
mkdir build
cd build
```

```bash
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_TIFF=ON -D BUILD_EXAMPLES=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -D CUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so --WITHFFMPEG …
```

### 5. 安装Gcc

```
sudo apt-get install gcc
```

```
gcc --version
```

### 6. 配置Darknet

#### 6.1 修改`darknet/Makefile`文件；

|          |          |
| -------- | -------- |
| CUDA =0  | CUDA =1  |
| CUDNN=0  | CUDNN=1  |
| OPENCV=0 | OPENCV=1 |
| AVX=0    | AVX=1    |
| OPENMP=0 | OPENMP=1 |
| LIBSO    | 1        |

#### 6.2 编译

执行

```
cd darknet

make
```

