# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json

classes=['균열 우수', '균열 보통', '균열 불량', 
        '박리,박락 우수', '박리,박락 보통', '박리,박락 불량', 
        '철근노출 우수', '철근노출 보통', '철근노출 불량', 
        '대지 우수', '대지 보통', '대지 불량', 
        '마감 우수', '마감 보통', '마감 불량', 
        '창호 우수', '창호 보통', '창호 불량', 
        '생활 우수', '생활 보통', '생활 불량']

def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode('utf8')
    return img_str

def detc(cfg, img, chk):
    rc = [] # threshold 넘은 class들 저장
    thr = 0.5
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, chk, device='cpu')
    # test a single image
    result = inference_detector(model, img)
    for c, l in enumerate(result):
        m = np.array(l)
        scores = m[...,4::5]
        for s in scores:
            if s[0] > thr:
                rc.append(classes[c])
    
    k = show_result_pyplot(
        model,
        img,
        result,
        palette='coco',
        score_thr=thr,
        out_file=None)

    kb64 = im_2_b64(k)
    res = dict()
    res["image"] = kb64
    res["class"] = rc
    return res