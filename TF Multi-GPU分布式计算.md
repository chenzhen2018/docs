# TF 计算加速

记录一些TF 在GPU上进行计算的知识点

## TF使用GPU

```python
import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0], name='a')
    b = tf.constant([3.0, 4.0], name='b')

with tf.device('/gpu:0'):
    c = a + b

config = tf.ConfigProto(log_device_placement=True)

with tf.Session(config=config) as sess:
    print(sess.run(c))
```

TF可以对不同的操作指定不同的计算设备，但是有些op不能使用GPU进行计算

可以通过**alow_soft_placement=True**自动将不能在GPU上计算的op放在CPU上计算

使用**config.gpu_options.per_process_gpu_memory_fraction=0.4**设置使用的显存

```
# 设定程序使用的计算设备
# method: 1. 
CUDA_VISIBLE_DEVICE=1 python test.py

# method: 2
import os
os.environ["CUDA_VISIBLE_DEVICE"] = "2"
```

GPU上计算密集型的运算，其他操作放在CPU上

另外，资源在GPU上的进入和转出都需要消耗时间

## 深度学习训练并行模式

>https://blog.csdn.net/qq_29462849/article/details/81185126

### 单机单卡

指的是在一张GPU上训练模型，更新参数的方式分为**整体梯度更新、随机梯度更新、batch梯度更新**

**batch梯度更新**指的是计算一个batch size数据的梯度，求取平均值，并根据该平均值来更新参数



### 单机多卡

**模型并行：**指的将模型的不同部分放在不同设备上进行训练

**数据并行：**指的是多个GPU使用相同的模型，但是每一次迭代的训练数据不一致。

数据并行模式中**参数更新的方式分为同步更新和异步更新**

#### 同步更新

优点：

1. loss下降比较稳定
2. 整个网络迭代步数增加，网络输入数据增加

缺点：

1. 出现短板效应，计算资源没有得到充分利用

#### 异步更新

优点：

1. 更新速度快，计算资源得到充分利用

缺点：

1. 出现过期梯度，loss下降不稳定，抖动大



