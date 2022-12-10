import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class MyDataset(CustomDataset):

    CLASSES = ('struct_crack_best', 'struct_crack_normal', 'struct_crack_faulty', 'struct_peel_best', 'struct_peel_normal', 'struct_peel_faulty', 'struct_rebar_best', 'struct_rebar_normal', 'struct_rebar_faulty', 'ground_best', 'ground_normal', 'ground_faulty', 'finish_best', 'finish_normal', 'finishi_faulty', 'window_best', 'window_normal', 'window_faulty', 'living_best', 'living_normal', 'living_faulty')

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue
        
            img_shape = ann_list[i+2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i+3])

            anns = ann_line.split(' ')
            bboxes = list()
            labels = list()
            for anns in ann_list[i+4:i+4+bbox_number]:
                bboxes.append([float(ann) for ann in anns[:4]])
                labels.append(int(anns[4]))

            data_infos.append(
                dict(
                    filename = ann_list[i+1],
                    width = width,
                    height = height,
                    ann = dict(
                        bboxes = np.array(bboxes).astype(np.float32),
                        labels = np.array(labels).astype(np.int64)
                    )
                )
            )
        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']