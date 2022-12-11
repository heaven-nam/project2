import os
import sys
import time
import numpy as np
# import imgaug
import skimage.draw

ROOT_DIR = os.path.abspath("../")
ANNOT_DIR = './validation/label/train'
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

############################################################
#  Configurations
############################################################

class MyConfig(Config):

    NAME = 'BuildingDefects'

    GPU_CONFIG = 1

    IMAGES_PER_GPU = 2

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 50

    NUM_CLASSES = 4 # 3 + 1(background)

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

############################################################
#  Dataset
############################################################

class BuildingDataset(utils.Dataset):

    def load_building(self,dataset_dir,subset):
        self.add_class('building',1,'Good')
        self.add_class('building',2,'Normal')
        self.add_class('building',3,'Bad')

        assert subset in ['train','validation']
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(ANNOT_DIR, 'annots.json')))
        annotations = list(annotations.values())

        annotations = [a for a in annotations if a['Annotations']]

        for a in annotations:
            for i in range(len(a)):
                bbox = a[i]['bbox']

        image_path = os.path.join(dataset_dir, a['Json_Data_ID'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        self.add_image(
            'building',
            image_id = a['Json_Data_ID'],
            path = image_path,
            width = width, height = height,
            bbox = bbox
        )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info['source'] != 'building':
            return super(self.__class__,self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['bbox'])], dtype = np.unit8)

        for i, b in enumerate(info['bbox']):
            r,c = skimage.draw.rectangle((bbox[0],bbox[3]),(bbox[2],bbox[1]))
            mask[r,c,i] = 1
        
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)