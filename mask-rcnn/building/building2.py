import os
import sys
import time
import numpy as np
import imgaug
import skimage.draw
import json
import dataframe_image as dfi
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
dataset_dir = '../'

from mrcnn.config import Config
from mrcnn import model as modellib, utils

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

############################################################
#  Configurations
############################################################

class MyConfig(Config):

    NAME = 'BuildingDefects'

    GPU_CONFIG = 1

    IMAGES_PER_GPU = 2

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 50

    NUM_CLASSES = 4 # 21 + 1(background)

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

############################################################
#  Dataset
############################################################

class BuildingDataset(utils.Dataset):

    def load_building(self,dataset_dir,subset):
        self.add_class('building',1,'균열_우수')
        self.add_class('building',2,'균열_보통')
        self.add_class('building',3,'균열_불량')
        self.add_class('building',4,'박리,박락_우수')
        self.add_class('building',5,'박리,박락_보통')
        self.add_class('building',6,'박리,박락_불량')
        self.add_class('building',7,'철근 노출_우수')
        self.add_class('building',8,'철근_노출_보통')
        self.add_class('building',9,'철근_노출_불량')
        self.add_class('building',10,'대지_우수')
        self.add_class('building',11,'대지_보통')
        self.add_class('building',12,'대지_불량')
        self.add_class('building',13,'마감_우수')
        self.add_class('building',14,'마감_보통')
        self.add_class('building',15,'마감_불량')
        self.add_class('building',16,'창호_우수')
        self.add_class('building',17,'창호_보통')
        self.add_class('building',18,'창호_불량')
        self.add_class('building',19,'생활_우수')
        self.add_class('building',20,'생활_보통')
        self.add_class('building',21,'생활_불량')
        

        assert subset in ['train','validation']
        if subset == 'train':
            json_dir = os.path.join(dataset_dir,'train_annotation.json')
            dataset_dir = os.path.join(dataset_dir,'train')
        else:
            json_dir = os.path.join(dataset_dir,'validation_annotation.json')
            dataset_dir = os.path.join(dataset_dir,'validation')
            
        annotations = json.load(open(json_dir, encoding='utf8'))
        annotations = list(annotations.values())

        annotations = [a for a in annotations]

        for a in annotations:
            b = a['Annotations']
            image_path = os.path.join(dataset_dir, a['Json_Data_ID'])
            # image = skimage.io.imread(image_path)
            height, width = 1440,1080

            self.add_image(
                source = 'building',
                image_id = a['Json_Data_ID'],
                path = image_path,
                width = width, height = height,
                annotations = b
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "building":
            return super(self.__class__, self).load_mask(image_id)
        
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]['annotations']

        for annotation in annotations:
            class_id = annotation['Class_ID']
            class_ids.append(class_id)
            m = np.zeros([info['height'], info['width'],1])
            bbox = annotation['bbox']
            xm, ym, xM, yM = bbox[0], bbox[1], bbox[2], bbox[3]
            start = (ym, xm)
            extent = (yM-ym, xM-xm)
            rr, cc = skimage.draw.rectangle(start, extent=extent, shape=m.shape)
            m[rr,cc,1] = 1
            instance_masks.append(m)
        
        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids

        # mask = np.zeros([info['height'], info['width'], len(info['bbox'])], dtype = np.uint8)
        # for i, bbox in enumerate(info['bbox']):
        #     xm,ym,xM,yM = bbox[0], bbox[1], bbox[2], bbox[3]
        #     xs = np.array([xm+1,xm+1,xM-1,xM-1])
        #     ys = np.array([ym-1,yM-1,yM-1,ym-1])
        #     rr, cc = skimage.draw.polygon(xs,ys)
        #     try:
        #         mask[cc,rr,i] = 1
        #     except:
        #         continue
        # # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        # num_ids = np.array(num_ids, dtype=np.uint8)
        # return mask.astype(bool), num_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'building':
            return info['path']
        else:
            super(self.__class__,self).image_reference(image_id)

def train(model):
    dataset_train = BuildingDataset()
    dataset_train.load_building(args.dataset, 'train')
    dataset_train.prepare()

    dataset_val = BuildingDataset()
    dataset_val.load_building(args.dataset, 'validation')
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs = 40,
                layers='heads')

def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image))*255
    
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= -1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)

    return splash

def detect_and(model, image_path=None):
    
    if image_path:
        image = skimage.io.imread(args.image)
        r = model.detect([image],verbose=1)[0]
        print(type(r), r.shape)
        print(r)

        splash = color_splash(image, r['masks'])

        file_name = "Test_1.png"
        skimage.io.imsave(file_name, splash)
    else:
        print('insert image')
    # skimage.io.imshow(image_path, splash)
    # skimage.io.show()



############################################################
#  Training
############################################################

if __name__ == '__main__':
    print('Start')
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Train Mask R-CNN to detect building defects'
    )
    parser.add_argument('command',
                        metavar='<command>')
    parser.add_argument('--dataset', required=False)
    parser.add_argument('--weights', required=False,
                        metavar='/path/to/weights.h5')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR)
    parser.add_argument('--image', required=False)
    args = parser.parse_args()

    if args.command == 'train':
        assert args.dataset, "Argument --dataset is requied for training"

    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print("Logs :", args.logs)

    if args.command == 'train':
        config = MyConfig()
    else:
        class InferenceConfig(MyConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir = args.logs)

    weight_path = args.weights

    if args.command == 'train':
        train(model)
    elif args.command == 'detect':
        detect_and(model, image_path = args.image)
    else:
        print('NO God NO!')