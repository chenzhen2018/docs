# docs

## 1. 当前内容

### 1. 1 Linux

[Ubuntu16.04下配置Darknet](./Ubuntu16.04下配置Darknet)

[Ubuntu16.04安装+CUDA+cuDNN+Anaconda+tensorflow-gpu](./Ubuntu16.04安装+CUDA+cuDNN+Anaconda+tensorflow-gpu.md)

[Ubuntu18.04下Object Detection API的使用](Ubuntu18.04下Object Detection API的使用.md)

[Ubuntu下远程连接相关](./远程连接)

[Ubuntu下Tenda U12无线网卡驱动安装](Ubuntu下Tenda U12驱动安装.md)

### 1.2 Windows



## 2. Pull

需要连接Fitow的网

```bash
git clone http://192.168.1.88:20080/ai/docs.git
```

输入用户名，密码



## 3. Push

### 3.1 Examples

上传

```
Ubuntu16.04下配置Darknet.md
```

操作:

```
cd docs
git add Ubuntu16.04下配置Darknet.md
git commit -m 'add Ubuntu16.04下配置Darknet'
git push origin master
```



## 4. 图片保存格式

最终所有文档中使用的图片都保存到`./data/`路径下，并根据文档名称重新生成一个同名文件夹，把文档使用的图片存入到相应的文件夹中

### 4.1 Typora配置

```
File-Preference-Images Insert,选择, Copy Image to custom folder

在输入栏中,键入:
./data/${filename}.assets
```

### 4.2 文档中使用图片

直接从原始图片位置进行复制，然后在`[]()`括号中进行粘贴，并在`[]`键入提示信息，相应图片会自动拷贝到`./data`中.



