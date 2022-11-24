#!/usr/bin/env python
# coding: utf-8

# <h1>&lt;&lt; 목차 &gt;&gt;<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#<<-Object-Detection-with-RetinaNet->>" data-toc-modified-id="<<-Object-Detection-with-RetinaNet->>-1">&lt;&lt; Object Detection with RetinaNet &gt;&gt;</a></span></li><li><span><a href="#Compute-IoU" data-toc-modified-id="Compute-IoU-2">Compute IoU</a></span></li><li><span><a href="#Anchor-Box" data-toc-modified-id="Anchor-Box-3">Anchor Box</a></span></li><li><span><a href="#RetinaNet-model" data-toc-modified-id="RetinaNet-model-4">RetinaNet model</a></span></li><li><span><a href="#Label-Encoder---중요" data-toc-modified-id="Label-Encoder---중요-5">Label Encoder - 중요</a></span></li><li><span><a href="#Preoporcessor" data-toc-modified-id="Preoporcessor-6">Preoporcessor</a></span></li><li><span><a href="#Generate-Dataset" data-toc-modified-id="Generate-Dataset-7">Generate Dataset</a></span></li><li><span><a href="#Classification-Loss-Function" data-toc-modified-id="Classification-Loss-Function-8">Classification Loss Function</a></span></li><li><span><a href="#Box-Regression-Loss-Function" data-toc-modified-id="Box-Regression-Loss-Function-9">Box Regression Loss Function</a></span></li><li><span><a href="#Transfer-Learning" data-toc-modified-id="Transfer-Learning-10">Transfer Learning</a></span></li><li><span><a href="#Prediction-Decoding" data-toc-modified-id="Prediction-Decoding-11">Prediction Decoding</a></span></li><li><span><a href="#Visualization-by-Heatmap-on-Image" data-toc-modified-id="Visualization-by-Heatmap-on-Image-12">Visualization by Heatmap on Image</a></span></li></ul></div>

# # << Object Detection with RetinaNet >>

# In[ ]:


import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


# In[ ]:



"""
## Downloading the COCO2017 dataset
Training on the entire COCO2017 dataset which has around 118k images takes a
lot of time, hence we will be using a smaller subset of ~500 images for
training in this example.
"""

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
keras.utils.get_file(filename, url)


with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")


# # Compute IoU

# In[ ]:




"""
## Implementing utility functions
Bounding boxes can be represented in multiple ways, the most common formats are:
- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions
`[x, y, width, height]`
Since we require both formats, we will be implementing functions for converting
between the formats.
"""


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.
    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0,
         boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


"""
## Computing pairwise Intersection Over Union (IOU)
As we will see later in the example, we would be assigning ground truth boxes
to anchor boxes based on the extent of overlapping. This will require us to
calculate the Intersection Over Union (IOU) between all the anchor
boxes and ground truth boxes pairs.
"""


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes
    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax


# # Anchor Box

# In[ ]:



"""
## Implementing Anchor generator
Anchor boxes are fixed sized boxes that the model uses to predict the bounding
box for an object. It does this by regressing the offset between the location
of the object's center and the center of an anchor box, and then uses the width
and height of the anchor box to predict a relative scale of the object. In the
case of RetinaNet, each location on a given feature map has nine anchor boxes
(at three scales and three ratios).
"""


class AnchorBox:
    """Generates anchor boxes.
    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.
    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self._areas = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]] 
        
                    #각 계층 앵커박스 넓이 (32*32=P3, 64*64=P4, 128*128=P5, 256*256=P6)
            
        self.scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        
                    #_areas로 정해진 넓이에서 3개의 scale로 3개의 다른 넓이 생성
                    # ex. 2^0=1배,  2^(1/3)=1.259배 , 2^(2/3)=1.587배
                
        self.aspect_ratios = [0.5, 1.0, 2.0]
        
                    # 앵커박스 종횡비 ( 0.5=1:2, 1.0=1:1, 2.0=2:1)
            
        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        
                    #한 지점에서 생성되는 좌표의 개수:9
                    #if len(self.aspect_ratios) * len(self.scales) = 3 * 3 = 9
                
        self._strides = [2**i for i in range(3, 8)]
        
                    #앵커박스 중앙점 계산위한 값(입력 영상에 대해 샘플링되는 비율)
                    # ex.2^3 = 8  -> P3에 대한 입력 영상에 비율
                    #    2^4 = 16 -> P4에 대한 입력 영상에 비율
                    #    2^5 = 32 -> P5에 대한 입력 영상에 비율
                    #    2^6 = 64 -> P6에 대한 입력 영상에 비율
                    #    2^7 = 128 -> P7에 대한 입력 영상에 비율
            
        self._anchor_dims = self._compute_dims()
        
                    # 멤버 메소드 _compute_dims() 결과값.
                    # 각 계층별 생성되는 9개의 앵커 박스를 list로 저장

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        즉, 해당 메소드는 5개의 각 계층별에 생성되는 9개의 앵커 박스를 생성한다.
        """
        anchor_dims_all = []
        for area in self._areas: # self._areas 각 계층 앵커박스 넓이 = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
            anchor_dims = []
            for ratio in self.aspect_ratios: #self.aspect_ratios 종횡비 = [0.5, 1.0, 2.0]
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:# self.scales = [2**x for x in [0, 1 / 3, 2 / 3]]
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all
    
    
    
#         (1) 구성 형태는 (1, 1, 9, 2)의 shape을 가지는 5개의 Tensor가 list에 저장되어 있다.
#                 첫 번째 Tensor는 P3에 대한 앵커 박스
#                 두 번째 Tensor는 P4에 대한 앵커 박스
#                 세 번째 Tensor는 P5에 대한 앵커 박스
#                 네 번째 Tensor는 P6에 대한 앵커 박스
#                 다섯 번째 Tensor는 P7에 대한 앵커 박스
        
#         (2) Tensor 구성
#             1. 종횡비 1:2의 1배 앵커 박스의 너비와 높이
#             2. 종횡비 1:2의 1.259배 앵커 박스의 너비와 높이
#             3. 종횡비 1:2의 1.587배 앵커 박스의 너비와 높이
#             4. 종횡비 1:1의 1배 앵커 박스의 너비와 높이
#             5. 종횡비 1:1의 1.259배 앵커 박스의 너비와 높이
#             6. 종횡비 1:1의 1.587배 앵커 박스의 너비와 높이
#             7. 종횡비 2:1의 1배 앵커 박스의 너비와 높이
#             8. 종횡비 2:1의 1.259배 앵커 박스의 너비와 높이
#             9. 종횡비 2:1의 1.587배 앵커 박스의 너비와 높이

    
    

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level
        Arguments:
        
            <해당 계층 feature 크기 & 레벨>
            
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.
        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
          
          
          해당 메소드는 한 계층에 대한 전체 앵커 박스의 좌표를 생성한다.
        """
        
        # (1) _compute_dims로 구한 _anchor_dims에서 해당 계층에 맞는 앵커 박스를 선택하여
        #      위에서 구한 각 중심 좌표과 매치시켜 (x, y, width, height) 형태로 구성한다.
        
#         예시) 너비과 높이이 2이고, level이 7인 feature map이 
#         (x,y, width, height)로 된 앵커 박스를 36개 생성 시
        
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        
        # centers (X_center, Y_center)
        
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        
        # dims : (Width, Height)
        
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        
        # anchors : (Xcenter, Ycenter, Width, Height)
        
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    
    
    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.
        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.
        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        
    # 해당 메소드는 ② _get_anchors 메소드를 이용하여
    # P3부터 P7까지의 모든 앵커 박스를 리스트에 append한 후에 Tensor로 변환시켜 반환한다.
    
    # ex. 입력 영상 크기 (256, 256)이면 각 계층 앵커박스 수는
    # P3 : 9216 | P4 : 2304 | P5 : 576 | P6 : 144 | P7 : 36
    #       전체 앵커 박수 수 : 12276 = 9216 + 2304 + 576 + 144 + 36
    
    # 반환하는 값의 shape은 (전체 앵커 박스 수, 4)
    
        
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2**i),
                tf.math.ceil(image_width / 2**i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # RetinaNet model

# In[ ]:


# 1. ResNet50 (Tensorflow의 keras에서 ImageNet pre=trained 된 ResNet50 backbone으로 가져오기)
def get_backbone():
    
    """Builds ResNet50 with pre-trained imagenet weights"""
    
    
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3] #, weights = 'imagenet'
    )
# input_shape=[None, None, 3] 즉, 입력 영상 높이와 너비 제한 없고
# 단 3채널 받기 - 사전학습 때 3채널 컬러영상 학습 했기 때문

    #backbone.trainable = trainable_flag
    #전이 학습 (transfer learning)시에 백본의 weights 값을 고정(freeze)여부를 선택하는 항목
    
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    
#    해당 함수의 목적은 사전 학습된 ResNet50에서 Conv 3, 4, 5계층에서 결괏값을 반환하는 것
    
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


"""
## Building Feature Pyramid Network as a custom layer
"""


class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.
    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
            """
#         해당 클래스는 (Figure 9-B)를 그대로 코드로 나타낸 것으로,
#         위의 A. def get_backbone() 함수를 통해 ResNet50을 backbone을 생성하고, 
#         이에 대해서 Conv 3, 4, 5계층의 결괏값을 받아서 Pyramid 3, 4, 5, 6, 7을 생성한다.
        


    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


"""
## Building the classification and box regression heads.
The RetinaNet model has separate heads for bounding box regression and
for predicting class probabilities for the objects. These heads are shared
between all the feature maps of the feature pyramid.
"""


def build_head(output_filters, bias_init):
    """
    해당 함수는 헤드를 생성한다. 입력 변수로 output_filters와 bias_init를 받는데,
    이들은 각각 헤드 마지막 레이어의 결과 필터 개수와 bias의 초깃값을 의미한다.
    
    Builds the class/box predictions head.
    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.
    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
        
        
        
        
    """
    
    # 5개의 Pyramid의 각 feature map 분류 헤드와 박스 regression 헤드가 붙어있는데, 
    # 각 feature map 크기가 모두 다르므로 입력 크기가 [None, None, 256]으로 설정
    
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        
        #stride =1, zero-padding, 256개의 3x3 conv과 ReLu 활성함수가 4번 반복
        
        head.add(
            keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters, 
            # 마지막으로 분류 헤드와 박스 regression 헤드의 필터 개수가 다르므로
            # 입력 변수로 output_filters를 받기
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            
            # 분류 헤드에서 bias_init으로 특정값으로 초기화하였는데,
            #[논문] 일반적으로 이진 분류에서는 각 클래스에 대해 동일한 확률로 초기화한다
            
        )
    )
    return head


"""
## Building RetinaNet using a subclassed model

분류 헤드에는 (9 * 클래스 개수)를 입력받고, 박스 regression 헤드에는 (9 * 4)를 입력으로 받는다.

"""


class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.
    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        # <<< prior probability. >>>am
        # Object detection에서는 배경 영역(Background)이 객체를 포함하는 영역(Foreground)보다 
        # 우세하기 때문에 학습에 편향이 생긴다
        # 이를 해결하기 위해 prior를 도입하여, 두 클래스에 대한 확률 모두를 0.01로 초기화하고자 한다. 
        #  본 코드에서는 -np.log((1 - 0.01) / 0.01)으로 설정되어있는데, 이는 softmax에 대해서 도출된 값
        
        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")
        
        # 박스 regression 헤드에서 초기화는 박스의 네 좌표를 모두 동일한 확률로 설정해도 상관없으므로,
        # “zero”를 초기화

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features: 
            
            # features = (P3 ~ P7의 feature map)
            #분류 feature는 클래스 개수만큼의 열로 변환(reshape)하고 밑으로 이어붙이고
            # 박스 좌표 feature는 좌표 수인 4열로 변환하고 이어붙인다
            
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)
        #  concat 함수를 통해 수 feature는 박스 feature 옆에 분류 feature를 이여
        #  최종 예측값(prediction value)를 도출
        # 결과 [X_center, Y_center, Width, Height, class_1, class_2, class_3, ...., class_N]
        # ex. 클래스 갯수(Num_classes)가 2일 때, 입력 영상이
        # RetinaNet의 모델을 거쳐 최종적으로 도출되는 예측값(prediction value) 형태이며
        # shape = (1, N, 6)을 가진다
        # 1은 Batch_size, N은 전체 앵커 박스 수, 6은 박스의 '4'개 좌표 + 클래스 갯수 '2'
        # 각 줄마다 각기 다른 앵커 박스의 4개 좌표와 각 클래스의 score가 포함되어있다.


# ![image.png](attachment:image.png)

# # Label Encoder - 중요

# In[ ]:


"""
## Encoding labels
The raw labels, consisting of bounding boxes and class ids need to be
transformed into targets for training. This transformation consists of
the following steps:
- Generating anchor boxes for the given image dimensions
- Assigning ground truth boxes to the anchor boxes
- The anchor boxes that are not assigned any objects, are either assigned the
background class or ignored depending on the IOU
- Generating the classification and regression targets using anchor boxes

RetinaNet model 코드 분석에서 def get_backbone():을 설명하면서
사전 학습된 모델을 사용하기에 입력 채널을 3채널 영상으로 맞춰야 한다고 하였다.

최종적으로 입력 데이터의 형태는 (1024, 1024, 1)에서 (512, 512, 3)으로 변경해야 한다.



"""


class LabelEncoder:
    """Transforms the raw labels into targets for training.
    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.
    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.
        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.
        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.
        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()


# # Preoporcessor

# In[ ]:



"""
## Preprocessing data
Preprocessing the images involves two steps:
- Resizing the image: Images are resized such that the shortest size is equal
to 800 px, after resizing if the longest side of the image exceeds 1333 px,
the image is resized such that the longest size is now capped at 1333 px.
- Applying augmentation: Random scale jittering  and random horizontal flipping
are the only augmentations applied to the images.
Along with the images, bounding boxes are rescaled and flipped if required.
"""


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance
    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.
    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`
    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.
    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def preprocess_data(sample):
    """Applies preprocessing step to a single sample
    Arguments:
      sample: A dict representing a single training sample.
    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # Generate Dataset

# ![image.png](attachment:image.png)

# # Classification Loss Function
# # Box Regression Loss Function

# In[ ]:


"""
## Implementing Smooth L1 loss and Focal Loss as keras custom losses
"""


class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference**2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)




class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss


# # Transfer Learning

# In[ ]:




"""
## Setting up training parameters
"""

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 80
batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

"""
## Initializing and compiling model
"""

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

"""
## Setting up callbacks
"""

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

"""
## Load the COCO2017 dataset using TensorFlow Datasets
"""

#  set `data_dir=None` to load the complete dataset

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

"""
## Setting up a `tf.data` pipeline
To ensure that the model is fed with data efficiently we will be using
`tf.data` API to create our input pipeline. The input pipeline
consists for the following major processing steps:
- Apply the preprocessing function to the samples
- Create batches with fixed batch size. Since images in the batch can
have different dimensions, and can also have different number of
objects, we use `padded_batch` to the add the necessary padding to create
rectangular tensors
- Create targets for each sample in the batch using `LabelEncoder`
"""

autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

"""
## Training the model
"""

# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch

epochs = 1

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

"""
## Loading weights
"""

# Change this to `model_dir` when not using the downloaded weights
weights_dir = "data"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

"""
## Building inference model
"""

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
int2str = dataset_info.features["objects"]["label"].int2str

for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

"""
Example available on HuggingFace.
| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Object%20Detection%20With%20Retinanet-black.svg)](https://huggingface.co/keras-io/Object-Detection-RetinaNet) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Object%20Detection%20With%20Retinanet-black.svg)](https://huggingface.co/spaces/keras-io/Object-Detection-Using-RetinaNet) |
"""


# # Prediction Decoding

# In[ ]:




"""
## Building the ResNet50 backbone
RetinaNet uses a ResNet based backbone, using which a feature pyramid network
is constructed. In the example we use ResNet50 as the backbone, and return the
feature maps at strides 8, 16 and 32.
"""

"""
## Implementing a custom layer to decode predictions
"""


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.
    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )