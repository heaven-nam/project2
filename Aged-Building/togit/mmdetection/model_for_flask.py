import mmcv
import mmdet
from demo import detc


# 2. test_model(image) 정의하기

def test_model(img_name):
        img = img_name
        config_file = 'swinretina_to_use.py'
        checkpoint_file = 'epoch_12.pth'
        #img = 'got_image/test333.jpg' #=img_name
        
        result = detc(config_file,img,checkpoint_file)
      
        return result

