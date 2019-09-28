### Mask RCNN

在Faster RCNN的基础上增加了一个分支，用于predicting an object mask



**Faster R-CNN**

1. stage one: RPN: propose candidate object bounding boxes
2. stage two: Faster R-CNN: extracts features using RoIPool from each candidate box,  and performs classification and bounding-box regression.

**Mask R-CNN**

1. stage one: RPN: 
2. stage two: predicting the class and box offset, and **Mask R-CNN also output a binary mask for each RoI.**



**Loss Function:**

对于each RoI
$$
L = L_{cls} + L_{box} + L_{mask}
$$
From Fast R-CNN:
$$
L_{cls}(p, u) = -\log p_u
$$
$p = \{p_0, ..., p_K\}$，表示当前RoI预测的K+1的类别概率，$p_u$表示对应true class u的概率
$$
L_{loc}(t^u, v) = \sum_{i \in \{x, y, w, h\}} smooth_{L_1}(t_{i}^u - v_i)
$$

$$
\begin{equation}
smooth_{L_1}(x) = \left\{
\begin{aligned}
0.5x^2  & & if |x| < 1 \\
|x|-0.5  & & otherwise \\
\end{aligned}
\right.
\end{equation}
$$

**Mask Branch Loss**

对于each RoI，mask分支有$Km^2$维输出，对于K个类别，生成K个$m\times m$大小的mask.
$$
L_mask = average \ binary \ cross-entropy \ loss(sigmoid(Mask_{k}^p), Mask_{k}^t)
$$
p表示预测，t表示ground truth， k表示真实标签是第k类，其他mask不参与计算。

**Mask branch会为每一个类别生成一个mask，但是会根据classification branch来选择输出的mask**

如果RoI与ground-truth box的IoU大于0.5，那么就将RoI看做是positive，只计算Positive RoI的损失.



**Mask Representation**

利用FCN对each RoI预测$m\times m$的mask，能够保证在mask branch分支中的每一层都能够保持其spatial layout



**RoIAlign**

RoIPool池化操作，会经过量化步骤，会有所损失。

具体地，下面描述RoIPool与RoIAlign

**RoIPool**

假如原图大小$800\times 800$，stride=32，那么得到的feature map大小$25 \times 25$

object大小$665 \times 665$，stride=32，那么最后$20.78 \times 20.78$, **会进行量化$20\times 20$**



假设最后想要得到$7\times 7$的输出，因此需要将$20 \times 20$的map分成49个块

因此，每一个小块的大小为$2.9 \times 2.9$，**量化后得到$2\times 2$**

因此，最后会得到49个$2 \times 2$的块。



**RoIAlign**

RoIAlign没有经过量化步骤，如下：

假如原图大小$800\times 800$，stride=32，那么得到的feature map大小$25 \times 25$

object大小$665 \times 665$，stride=32，那么最后$20.78 \times 20.78$



假设最后想要得到$7\times 7$的输出，因此需要将的$20.78 \times 20.78$map分成49个块

因此，每一个小块的大小为$2.97 \times 2.97$

对$2.97 \times 2.97$的小块分成四份，每一份利用线性插值方法得到数值后，再对四小份执行max pool/average pool



**Network Architecture**

**backbone**

ResNet, ResNeXt networks of depth 50 or 101 layers

FPN

**head**

延续ResNet-C4与FPN，增加了mask branch