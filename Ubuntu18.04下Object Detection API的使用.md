### Ubuntu下Object Detection API的使用

配置信息：

```
Ubuntu 18.04
NVIDIA GTX 1080Ti
CUDA 10.0
cuDNN 7.5
```

关于显卡驱动、CUDA、cuDNN默认已正确配置；

### 1. 配置

可以参考： https://blog.csdn.net/qq_28019591/article/details/82023506



大部分命令行操作是在./models/research/进行的

下载models: https://github.com/tensorflow/models

使用pycharm指定python环境，以及相应的第三方库

```
tensorflow-gpu
pillow
matplotlib
```

#### COCO API installation

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools models/research/
```

#### Protobuf Compilation

下载3.4.0版本的protobuf compilation: https://github.com/protocolbuffers/protobuf/tags

解压完成后，

```
sudo cp protoc-3.4.0-linux-x86_64/bin/protoc /usr/bin/protoc
```

```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

就可以在models/research/object_detection/protos/下看到同名的py文件

#### Add Libraries to PYTHONPATH

```
cd models/research/slim
python setup.py build
python setup.py install
```

```
sudo gedit ~/.bashrc

# 在最下面加入，其中pwd表示slim文件夹所在的路径
export PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim

# 使修改生效
source ~/.bashrc
```

```
# 测试上述操作
python 
import slim
# 导入成功，即可
```

#### 配置测试

```
cd models/research
pthon object_detection/builders/model_builders_test.py
# 打印OK表明成功
```

### 2. 测试

使用提供的训练好的模型进行测试

下载模型：http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz

解压到models/research/object_detection/下，得到ssd_mobilenet_v1_coco_2017_11_17文件夹

在models/research/object_detection/下新建文件cz_inference.py，将下面的代码添加进去

```python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util

import cv2

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# # This is needed to display the images.
# %matplotlib inline


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'pascal_label_map.pbtxt')
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def download_pre_model():
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


def main():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    for image_path in TEST_IMAGE_PATHS:
        print(image_path)
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        # print(image_np.shape, type(image_np))
        # plt.figure(figsize=IMAGE_SIZE)
        # plt.axis('off')
        # plt.imshow(image_np)
        # plt.show()

        cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)


if __name__ == '__main__':
    # download_pre_model()
    main()
```



之后执行

```
cd models/research/object_detection/
python cz_inference.py

# 预测两张图片
```

如果想要使用其他训练好的模型文件，在https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md进行下载，并在cz_inference.py文件中修改相应位置即可；



### 3. 训练

下载pascal voc 2012数据集，然后转换成tfrecord文件，修改配置文件，进行训练；（可以使用预训练模型，也可以从头训练）

##### [Generating the PASCAL VOC TFRecord files](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md)

修改models/research/object_detection/dataset_tools/create_pascal_tfrecord.py文件

```
main方法中，
examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                            'aeroplane_' + FLAGS.set + '.txt')
修改成
examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                 FLAGS.set + '.txt')
```

对下面代码进行相应修改，执行：

```python
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

#### 准备训练

在models/research/object_detection/建立下面文件夹

```
./object_detection
	chen/
		data/
		models/
			test_0711/
				model/
				pre_model/
				result/
				ssd_mobilenet_v1_coco.config
```

chen/：用于存放训练过程中所有数据

data/：用于存放上述步骤中生成的tfrecord数据、pbtxt文件（/models/research/data/中有一些）

models: 用于存储模型与结果的地方

test_0711: 表示此次训练的总文件夹

model: 用于存放训练的模型

pre_model: 用于存放预训练的模型

result: 用于存放训练结束后的pb模型文件

ssd_mobilenet_v1_coco.config: 训练的配置文件，主要作用是指明训练使用的网络结构、数据、预训练模型位置、模型存储位置、以及一些超参数

#### .config配置

其中需要更改的有：

类别个数：

batch_size:

预训练模型位置

训练集、验证集数据、pbtxt文件

一些超参数的设置

#### 修改model_main.py

或者执行命令的时候指定参数也行

主要修改的地方是：模型保存的位置、以及config的位置

```
cd models/research/
python object_detection/model_main.py
```

#### 固化模型

修改/model/research/object_detection/export_inference_graph.py文件

```
cd models/research/
python object_detection/export_inference_graph.py --pipeline_config_path ** --trained_checkpoint_prefix ** --output_directory **

# 或者在文件中进行修改
```

