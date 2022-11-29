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

import mrcnn

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_instances

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

############################################################
#  Configurations
############################################################

class MyConfig(Config):

    NAME = 'BuildingDefects'

    GPU_CONFIG = 1

    IMAGES_PER_GPU = 2

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 50

    NUM_CLASSES = 22 # 21 + 1(background)

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

############################################################
#  Dataset
############################################################

class BuildingDataset(utils.Dataset):

    def load_building(self,dataset_dir,subset):
        # self.add_class('building',1,'Good')
        # self.add_class('building',2,'Normal')
        # self.add_class('building',3,'Bad')
        self.add_class('building',1,'Crack_Good')
        self.add_class('building',2,'Crack_Normal')
        self.add_class('building',3,'Crack_Bad')
        self.add_class('building',4,'Spalling_Good')
        self.add_class('building',5,'Spalling_Normal')
        self.add_class('building',6,'Spalling_Bad')
        self.add_class('building',7,'exposure of rebar_Good')
        self.add_class('building',8,'exposure of rebar_Normal')
        self.add_class('building',9,'exposure of rebar_Bad')
        self.add_class('building',10,'Ground_Good')
        self.add_class('building',11,'Ground_Normal')
        self.add_class('building',12,'Ground_Bad')
        self.add_class('building',13,'finish_Good')
        self.add_class('building',14,'finish_Normal')
        self.add_class('building',15,'finish_Bad')
        self.add_class('building',16,'Window_Good')
        self.add_class('building',17,'Window_Normal')
        self.add_class('building',18,'Windows_Bad')
        self.add_class('building',19,'Living_Good')
        self.add_class('building',20,'Living_Normal')
        self.add_class('building',21,'Living_Bad')
        

        assert subset in ['train','validation']
        if subset == 'train':
            json_dir = os.path.join(r'../','train_annotation.json')
            dataset_dir = os.path.join(dataset_dir,'train')
        else:
            json_dir = os.path.join(r'../','validation_annotation.json')
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
            m = np.zeros([info['height'], info['width']])
            bbox = annotation['bbox']
            xm, ym, xM, yM = bbox[0], bbox[1], bbox[2], bbox[3]
            start = (ym, xm)
            extent = (yM-ym, xM-xm)
            rr, cc = skimage.draw.rectangle(start, extent=extent, shape=m.shape)
            m[rr,cc] = 1
            instance_masks.append(m)
        
        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids

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
                epochs = 30,
                layers='all')

def detect_and(model, image_path=None):
    image = skimage.io.imread(args.image)

    # class_names
    class_names = ['BG','Crack_Good','Crack_Normal','Crack_Bad','Spalling_Good','Spalling_Normal','Spalling_Bad','exposure of rebar_Good','exposure of rebar_Normal','exposure of rebar_Bad','Ground_Good','Ground_Normal','Ground_Bad','finish_Good','finish_Normal','finish_Bad','Window_Good','Window_Normal','Window_Bad','Living_Good','Living_Normal','Living_Bad']
    # class_names = ['BG','Good','Normal','Bad']

    if image_path:
        image = skimage.io.imread(args.image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        r = model.detect([image],verbose=0)
        r = r[0]
        print(r.keys())

        mrcnn.visualize.display_instances(image=image,
                                          class_names=class_names,
                                          boxes=r['rois'],
                                          masks = r['masks'],
                                          class_ids=r['class_ids'],
                                          scores=r['scores'])
    else:
        print('insert image')



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