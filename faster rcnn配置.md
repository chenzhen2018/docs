### Faster RCNN配置



#### Installation

```
git clone https://github.com/endernewton/tf-faster-rcnn.git
```

```
cd tf-faster-rcnn/lib

# 修改GPU -arch
vim setup.py
```

```
cd tf-faster-rcnn/lib
make clean
make 
cd ..
```

```
# 安装Python COCO API
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```

#### Setup data

Pascal VOC 2007

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

```bash
mkdir ./data/VOCdevkit2007/

将VOCdevkit/VOC2007下所有文件夹拷贝到上述路径
```

**Download pre-trained model**

```
./data/scripts/fetch_faster_rcnn_models.sh

# 或
https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ
下载res101下的voc_0712_80k-110k.tgz
```

```bash
cd tf-faster-rcnn
mkdir -p output/res101/voc_2007_trainval+voc_2012_trainval
cd output/res101/voc_2007_trainval+voc_2012_trainval
ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default

cd ../../..
```

#### Test some images

```bash
suo gedit ./tools/demo.py
```

替换成使用opencv作图

```python
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.imshow('img', im)
    cv2.waitKey(0)
```

```python
python ./tools/demo.py
```

#### 使用预训练模型测试

```python
GPU_ID=0
./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
```

**修改voc_eval.py**

```
# 1.
cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
修改成
cachefile = os.path.join(cachedir, 'annots.pkl')

# 2. 123
with open(cachefile, 'w') as f:
修改成
with open(cachefile, 'wb') as f:
```

#### Train your own model

下载预训练模型

```bash
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..
```

```python
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
```





