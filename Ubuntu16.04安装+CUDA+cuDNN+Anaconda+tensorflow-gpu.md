配置信息

```
Windows 10
RTX 2080Ti


Ubuntu 16.04
```





#### 1. windows下分配空间

在磁盘管理中压缩卷200G，会得到一个200G为未分配的空间

#### 2. 启动盘制作

#### 3. 设置U盘启动盘启动

重启电脑，长按Del建，打开bios；
设置，Advance > CSM Configuration > Boot option filter 为 **UEFT and Legacy**
设置，Boot > Hard Drive BBS Proiorities，将盘设置为最高优先级

退出，保存设置


#### 4.安装Ubuntu 16.04
进入Ubutnu安装界面，选择 Install Ubuntu

Welcom: English

Preparing to install Ubuntu: 不勾选

Installation type: something else

空间分配：

|           |        |
| --------- | ------ |
| /         | 100G   |
| swap area | 16G    |
| /boot     | 1G     |
| /home     | 剩下的 |

**注意：**

**将下面的Device for boot loader installation:选择为boot所在的位置**

选择Install Now

设置地点、语言、计算机用户名、密码，开始安装；



安装完成，提示重启，重启后，拔掉U盘；



#### 5. 解决low-graphics mode 错误问题

重启成功后，会提示进入Ubuntu（这时没有进入Win10）的选项，选择Ubuntu进入，

会提示**The system is running in low-graphics mode**问题；一路点击OK

会提示/dev/nvme0n1p3:... 错误



执行下面命令进入命令行：

```
Ctrl+Alt+F1
```

进入命令行后，执行下面命令

```
cd /etc/X11
sudo cp xorg.conf.failsafe xorg.conf
```

之后，重启

```
reboot
```

重启，选择Ubuntu启动（仍没有Win10启动选项），输入密码，进入系统；

```
sudo apt-get update
sudo apt-get upgrade
```

重启，**这时就会有启动Win10的选项了**



#### 6. 安装nvidia驱动

上一步中最后一步重启，选择Ubuntu启动，需要做两件事情

1. 禁用nouveau
2. 下载驱动程序

##### 6.1 下载驱动程序

在nvidia官网下载，适合显卡的驱动程序，**并将驱动文件拷贝到/Downloads/目录下；**

```
NVIDIA-Linux-x86_64-430.34.run
```

拔掉硬盘；

##### 6.2 禁用Nouveau

```shell
sudo gedit /etc/modprobe.d/blacklist.conf
```

在打开的文件最后一行，添加：

```
blacklist nouveau
options nouveau modeset=0
```

```bash
# 执行
update-initramfs –u
```

```shell
# 重启后执行，无输出则表明禁用nouveau成功
lsmod | grep nouveau
```

##### 6.3 安装驱动程序

命令行执行，目的是为了禁用图形界面服务

```shell
sudo service lightdm stop
```

跳出黑框，输入：

```shell
Ctrl+Alt+F1 
```

进入命令行；

输入用户名、密码后，进入系统；

执行：

```shell
cd Downloads
sudo chmod a+x NVIDIA-Linux-x86_64-430.34.run
```

接着，输入

```shell
sudo ./NVIDIA-Linux-x86_64-410.78.run –no-opengl-files –no-x-check –no-nouveau-check
```

回车，开始安装驱动；**全是Yes**

成功后，会回到命令行；

输入下面命令，**启动图像界面服务**

```shell
sudo service lightdm start
```

这时，会自动进入图形界面，如果不自动进入，可以按`Ctrl+Alt+F7`；

输入密码，进入系统后

```shell
nvidia-smi
```

输出显卡信息，表明驱动安装成功；

#### 7. 配置CUDA9.0+cuDNN7.0

下载CUDA9.0, cuDNN7.0，文件会在Download文件夹下；

##### 7.1 安装CUDA9.0

执行

```shell
cd Downloads
sudo sh cuda_9.0.176_384.81_linux.run
```

按`Ctrl+C`，然后输入`accept`，

提示是否安装驱动，选择`n`;

后面的选择**yes**，或者**默认**;

```shell
sudo gedit ~/.bashrc
```

在打开的文件中的最后，输入

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```

保存，执行

```shell
source ~/.bashrc
```

执行

```shell
nvcc -V
```

输出CUDA版本信息

或：

```shell
cat /usr/local/cuda/version.txt
```

##### 7.2 配置cuDNN

将下载的cuDNN文件夹解压得到cuda文件夹，执行

```shell
cd Downloads

sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include/
 
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
 
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h
 
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

执行

```shell
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

输出cuDNN信息，则表明配置成功；

##### 7.3 切换CUDA+cuDNN版本

```shell
sudo rm -rf cuda
sudo ln -s /usr/local/cuda-10.0 usral/cuda
```

查看cuda指向哪个版本的CUDA

```shell
cd /usr/local/

stat cuda
```



#### 8 安装Anaconda

执行

```shell
bash Anaconda3-5.0.1-Linux-x86_64.sh
```

最后一步，将路径添加到/.bashrc，**输入yes**；

重新打开一个terminal，输入`python`，会输出Anaconda的python；

##### 8.1 安装Tensorflow-gpu版本

**更换成阿里云源**

```shell
cd /etc/apt/

sudo cp sources.list sources.list.bak
```

打开：

```shell
sudo gedit sources.list
```

使用下面内容替换原有内容：

```
 deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse  
 deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse 
 deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse 
 deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse  
 ##测试版源  
 deb http://mirrors.aliyun.com/ubuntu/ xenial-proposed main restricted universe multiverse 
 # 源码  
 deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse  
 deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse  
 deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse  
 deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse  
 ##测试版源  
 deb-src http://mirrors.aliyun.com/ubuntu/ xenial-proposed main restricted universe multiverse  
 # Canonical 合作伙伴和附加  
 deb http://archive.canonical.com/ubuntu/ xenial partner  
 deb http://extras.ubuntu.com/ubuntu/ xenial main  
```

保存后执行：

```shell
sudo apt-get update
sudo apt-get upgrade
```



之后安装tensorflow-gpu

```shell
pip install tensorflow-gpu==1.10.0
```



安装成功后测试：

```python
python

import tensorflow as tf

print(tf.test.gpu_device_name())
```



